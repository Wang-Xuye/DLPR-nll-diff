import os
import argparse

_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--gpu-id", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
_bootstrap_args, _ = _bootstrap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _bootstrap_args.gpu_id

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from utils.data.transform import build_transforms

import numpy as np
import math
from tqdm import tqdm

from nll_model import NearLosslessCompressor, RateDistortion

from global_nii_max import nii_max, nii_max_f


class FrameDataset(data.Dataset):
    """Flatten 2D images and 3D medical-image sequences into single frames."""

    def __init__(self, path_dir, transform=None):
        self.path_dir = path_dir
        self.transform = transform
        self.samples = []

        files = sorted(
            os.path.join(path_dir, name)
            for name in os.listdir(path_dir)
            if name.lower().endswith(".npy")
        )
        if not files:
            raise FileNotFoundError(f"No .npy files found under: {path_dir}")

        for path in files:
            shape = np.load(path, mmap_mode="r").shape
            if len(shape) == 2:
                self.samples.append((path, None, None))
            elif len(shape) == 3:
                frame_axis = int(np.argmin(shape))
                spatial_shape = [shape[i] for i in range(3) if i != frame_axis]
                if shape[frame_axis] >= min(spatial_shape):
                    raise ValueError(
                        f"Cannot infer frame dimension for {path}; shape={shape}."
                    )
                for frame_idx in range(shape[frame_axis]):
                    self.samples.append((path, frame_axis, frame_idx))
            else:
                raise ValueError(
                    f"Expected a 2D image or 3D sequence, got shape={shape} in {path}."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, frame_axis, frame_idx = self.samples[index]
        img = np.load(path)
        if frame_axis is not None:
            img = np.take(img, frame_idx, axis=frame_axis)
        img = np.asarray(img, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"Expected one 2D frame, got shape={img.shape} from {path}")
        if self.transform is not None:
            img = self.transform(img)
        return img


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_model_state(model, state_dict):
    target_model = unwrap_model(model)
    model_keys = set(target_model.state_dict().keys())
    candidate = state_dict
    if not model_keys.intersection(candidate.keys()) and any(k.startswith("module.") for k in candidate.keys()):
        candidate = strip_module_prefix(candidate)
    elif model_keys and all(("module." + k) in candidate for k in model_keys):
        candidate = strip_module_prefix(candidate)
    target_model.load_state_dict(candidate)


def configure_optimizers(model, lr, aux_lr):

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and not n.startswith("residual_compressor_cond.") and not n.startswith("fusion_cond.") and p.requires_grad
    }
    parameters_nll = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and (n.startswith("residual_compressor_cond.") or n.startswith("fusion_cond."))
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params1 = parameters & aux_parameters
    inter_params2 = parameters_nll & aux_parameters
    inter_params3 = parameters & parameters_nll
    union_params = parameters | parameters_nll | aux_parameters

    assert len(inter_params1) == 0
    assert len(inter_params2) == 0
    assert len(inter_params3) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    optimizer_nll = optim.Adam(
        (params_dict[n] for n in sorted(parameters_nll)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )

    return optimizer, aux_optimizer, optimizer_nll


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, optimizer_nll, train_step, epoch, tb_writer=None, clip_max_norm=None):
    
    model.train()
    device = next(model.parameters()).device
    core_model = unwrap_model(model)
    
    train_size = 0
    progress = tqdm(
        train_dataloader,
        desc=f"Train epoch {epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
    )
    for x in progress:
        x = x.to(device).contiguous()
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        optimizer_nll.zero_grad()
        
        out = model(x)
        
        out_criterion = criterion(out, x)
        out_criterion["loss"].backward()
        if clip_max_norm:
            nn.utils.clip_grad_norm_(core_model.parameters_wo_cond(), clip_max_norm)
        optimizer.step()

        out_criterion["res_bpp_cond"].backward()
        if clip_max_norm:
            nn.utils.clip_grad_norm_(core_model.parameters_w_cond(), clip_max_norm)
        optimizer_nll.step()
        
        aux_loss = core_model.lossy_compressor.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        train_step += 1
        if tb_writer and train_step % 10==1:
            tb_writer.add_scalar('train loss', out_criterion["loss"].item(), train_step)
            tb_writer.add_scalar('train mse', out_criterion["mse_loss"].item(), train_step)
            tb_writer.add_scalar('train img bpp', out_criterion["img_bpp"].item(), train_step)
            tb_writer.add_scalar('train res bpp', out_criterion["res_bpp"].item(), train_step)
            tb_writer.add_scalar('train res bpp cond', out_criterion["res_bpp_cond"].item(), train_step)
            tb_writer.add_scalar('train bpp', out_criterion["img_bpp"].item() + out_criterion["res_bpp"].item(), train_step)
            tb_writer.add_scalar('train aux', core_model.lossy_compressor.aux_loss().item(), train_step)
            
        train_size += x.shape[0]
        progress.set_postfix(
            loss=f"{out_criterion['loss'].item():.4f}",
            bpp=f"{(out_criterion['img_bpp'] + out_criterion['res_bpp']).item():.4f}",
        )
        
    print("train sz:{}".format(train_size))
    return train_step
            
        
def eval_epoch(model, criterion, eval_dataloader, epoch, tb_writer=None):
    
    model.eval()
    device = next(model.parameters()).device
    core_model = unwrap_model(model)
    
    loss = 0
    img_bpp = 0
    res_bpp = 0
    res_bpp_cond = 0
    mse_loss = 0
    aux_loss = []
    
    if tb_writer:
        save_imgs = True
    else:
        save_imgs = False
    
    
    eval_size = 0
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to(device).contiguous()
            
            out = model(x)
            out_criterion = criterion(out, x)

            N, _, H, W = x.shape
            
            loss += out_criterion["loss"] * N 
            img_bpp += out_criterion["img_bpp"] * N 
            res_bpp += out_criterion["res_bpp"] * N
            res_bpp_cond += out_criterion["res_bpp_cond"] * N
            mse_loss += out_criterion["mse_loss"] * N
            aux_loss.append(core_model.lossy_compressor.aux_loss())
            
            if save_imgs:
                x_rec = out["x_hat"].clamp_(0, nii_max)
                tb_writer.add_image('input/0', x[0,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('input/1', x[1,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('input/2', x[2,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('output/0', x_rec[0,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('output/1', x_rec[1,:,:,:].to(torch.uint8), epoch)
                tb_writer.add_image('output/2', x_rec[2,:,:,:].to(torch.uint8), epoch)
                save_imgs = False
            
            eval_size += N
            
        loss = (loss/eval_size).item()
        img_bpp = (img_bpp/eval_size).item()
        res_bpp = (res_bpp/eval_size).item()
        res_bpp_cond = (res_bpp_cond / eval_size).item()
        bpp_loss = img_bpp + res_bpp
        mse_loss = (mse_loss/eval_size).item()
        aux_loss = (sum(aux_loss)/len(aux_loss)).item()
        psnr = 10. * np.log10(nii_max_f**2/mse_loss)
        if tb_writer:
            tb_writer.add_scalar('eval/eval loss', loss, epoch)
            tb_writer.add_scalar('eval/eval bpp', bpp_loss, epoch)
            tb_writer.add_scalar('eval/eval img bpp', img_bpp, epoch)
            tb_writer.add_scalar('eval/eval res bpp', res_bpp, epoch)
            tb_writer.add_scalar('eval/eval res bpp cond', res_bpp_cond, epoch)
            tb_writer.add_scalar('eval/eval mse', mse_loss, epoch)
            tb_writer.add_scalar('eval/eval psnr', psnr, epoch)
            tb_writer.add_scalar('eval/eval aux', aux_loss, epoch)
            
        print("eval sz:{}".format(eval_size))
    
    return loss, bpp_loss, img_bpp, res_bpp, res_bpp_cond, mse_loss, psnr, aux_loss
        

def train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir, resume_path=""):
    parser = argparse.ArgumentParser(description="DLPR baseline training")
    parser.add_argument("--train-path", type=str, default="../Mosmed/train-s", help="Path to training slices")
    parser.add_argument("--eval-path", type=str, default="../Mosmed/test-s", help="Path to eval slices")
    parser.add_argument("--ckp-dir", type=str, default="./ckp_nll/dlpr_baseline", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs_nll/dlpr_baseline", help="TensorBoard/log directory")
    parser.add_argument("--epochs", type=int, default=600, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--resume-path", type=str, default="", help="Optional checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu-id", type=str, default=_bootstrap_args.gpu_id, help="Visible GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--parallel", type=str, default="auto", choices=["auto", "none", "data_parallel"], help="Parallel training mode")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.ckp_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    transform_train = build_transforms("p64")
    transform_eval = build_transforms("p64_centercrop")
    train_data = FrameDataset(args.train_path, transform=transform_train)
    train_dataloader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    eval_data = FrameDataset(args.eval_path, transform=transform_eval)
    eval_dataloader = data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    nll_module = NearLosslessCompressor(192, 5, "3P").cuda()
    use_parallel = args.parallel == "data_parallel" or (args.parallel == "auto" and torch.cuda.device_count() > 1)
    if use_parallel:
        nll_module = nn.DataParallel(nll_module)

    rd_criterion = RateDistortion(lmbda=0.001)
    optimizer, aux_optimizer, optimizer_nll = configure_optimizers(unwrap_model(nll_module), lr=1e-4, aux_lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True)
    scheduler_nll = optim.lr_scheduler.MultiStepLR(optimizer_nll, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True)

    tb_writer = SummaryWriter(args.log_dir)

    if args.resume_path:
        ckp = torch.load(args.resume_path, map_location="cpu")
        start_epoch = ckp['epoch'] + 1
        load_model_state(nll_module, ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer_state_dict'])
        optimizer_nll.load_state_dict(ckp['optimizer_nll_state_dict'])
        aux_optimizer.load_state_dict(ckp['aux_optimizer_state_dict'])
        scheduler.load_state_dict(ckp['scheduler'])
        scheduler_nll.load_state_dict(ckp['scheduler_nll'])
        train_step = ckp['step']
    else:
        start_epoch = 0
        train_step = 0

    for epoch in range(start_epoch, args.epochs):
        # train
        train_step = train_one_epoch(nll_module, rd_criterion, train_dataloader, optimizer, aux_optimizer, optimizer_nll, train_step, epoch, tb_writer, clip_max_norm=1.0)
        # eval
        loss, bpp_loss, img_bpp, res_bpp, res_bpp_cond, mse_loss, psnr, aux_loss = eval_epoch(nll_module, rd_criterion, eval_dataloader, epoch, tb_writer)

        scheduler.step()
        scheduler_nll.step()

        # Save at the configured interval and always keep the final epoch.
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unwrap_model(nll_module).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_nll_state_dict': optimizer_nll.state_dict(),
                'aux_optimizer_state_dict': aux_optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler_nll': scheduler_nll.state_dict(),
                'step': train_step,
            }
            torch.save(checkpoint, os.path.join(args.ckp_dir, "ckp.tar"))
            torch.save(
                checkpoint,
                os.path.join(args.ckp_dir, f"ckp-Ep{epoch + 1:04d}.tar"),
            )
            print(f"Checkpoint saved: epoch {epoch + 1}")

        log_str = "Epoch(Eval):{}, bpp:{}, img bpp:{}, res bpp:{}, res bpp cond:{}, mse:{}, psnr:{}".format(
            epoch, bpp_loss, img_bpp, res_bpp, res_bpp_cond, mse_loss, psnr
        )
        
        print(log_str)  # 终端打印
        
        with open(os.path.join(args.log_dir, "train.log"), "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
    tb_writer.close()


if __name__ == "__main__":
    train(None, None, None, None, None)


r'''
python train.py \
  --gpu-id 2 \
  --parallel none \
  --train-path ../Mosmed/train-slice \
  --eval-path ../Mosmed/test-slice \
  --ckp-dir ./ckp_nll/dlpr_baseline \
  --log-dir ./logs_nll/dlpr_baseline \
  --epochs 20000 \
  --batch-size 64 \
  --num-workers 8 \
  --save-every 50

python train.py \
  --gpu-id 0,1,2,3 \
  --parallel auto \
  --train-path ../Mosmed/train-slice \
  --eval-path ../Mosmed/test-slice \
  --ckp-dir ./ckp_nll/dlpr_baseline \
  --log-dir ./logs_nll/dlpr_baseline \
  --epochs 20000 \
  --batch-size 64 \
  --num-workers 8 \
  --save-every 50
'''