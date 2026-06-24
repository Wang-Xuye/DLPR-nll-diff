import os
import sys


def read_cuda_visible_devices(default="3"):
    for idx, arg in enumerate(sys.argv):
        if arg in ("--gpu-id", "--cuda-visible-devices") and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--gpu-id="):
            return arg.split("=", 1)[1]
        if arg.startswith("--cuda-visible-devices="):
            return arg.split("=", 1)[1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", default)


CUDA_VISIBLE_DEVICES = read_cuda_visible_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ['TMPDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(os.environ['TMPDIR'], exist_ok=True)

# ============ 路径配置 ============
SAVE_DIR_DEFAULT = "log4_diff_warmup_nobpp200"
# =================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from utils.data.datasets import ImageDataset
from utils.data.transform import build_transforms
import random
import numpy as np
import math
from tqdm import tqdm

from nll_model import NearLosslessCompressor, RateDistortion
from nll_model_p import NearLosslessCompressor_I, NearLosslessCompressor_P, RateDistortion, unpatchify_from_batch, patchify_to_batch, LossyCompressor_P_diff_sit
from global_nii_max import nii_max, nii_max_f


def replicate_pad(x, p):
    height = x.shape[-2]
    width = x.shape[-1]
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    if padding_right == 0 and padding_bottom == 0:
        return x
    return np.pad(x, [(0, 0), (0, padding_bottom), (0, padding_right)], mode='edge')


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


class Dataset_seq():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                clip = np.load(full_path).transpose(2, 0, 1)
                self.clips.append(clip)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip0 = self.clips[index]
        clip1 = np.expand_dims(clip0, axis=1)

        T, C, H, W = clip1.shape
        start_idx = np.random.randint(0, T - 8)
        clip1 = clip1[start_idx:start_idx + 8, :, :, :]
        if random.random() < 0.5:
            clip1 = clip1[:, :, :, ::-1]
        if random.random() < 0.5:
            clip1 = clip1[:, :, ::-1, :]
        clip1 = np.ascontiguousarray(clip1, dtype=np.float32)
        return clip1


class Dataset_test():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                clip = np.load(full_path)
                clip = replicate_pad(clip, 128)
                self.clips.append(clip)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip0 = self.clips[index]
        clip1 = np.expand_dims(clip0, axis=1)
        clip1 = np.ascontiguousarray(clip1, dtype=np.float32)
        return clip1


class RateDistortion_diff(nn.Module):
    """Modified RateDistortion that uses diffusion_loss instead of MSE."""
    def __init__(self, lmbda=0.1, use_bpp=True, lmbda_diff_lmm=0.05):
        super().__init__()
        self.lmbda = lmbda
        self.use_bpp = use_bpp
        self.lmbda_diff_lmm = lmbda_diff_lmm
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["img_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["img_likelihoods"].values()
        )
        out["res_bpp"] = output["res_likelihoods"].sum() / (-math.log(2) * num_pixels)
        out["res_bpp_cond"] = output["res_likelihoods_cond"].sum() / (-math.log(2) * num_pixels)

        if "diffusion_loss" in output:
            out["mse_loss"] = output["diffusion_loss"]
        else:
            x_hat = torch.clamp(output["x_hat"], min=-0.5, max=nii_max_f + 0.5)
            out["mse_loss"] = self.mse(x_hat, target)

        out["diff_lmm_loss"] = output.get("diff_lmm_loss",
                                          torch.zeros((), device=target.device))

        if self.use_bpp:
            out["loss"] = (
                self.lmbda * out["mse_loss"]
                + out["img_bpp"]
                + out["res_bpp"]
                + self.lmbda_diff_lmm * out["diff_lmm_loss"]
            )
        else:
            out["loss"] = out["mse_loss"] + self.lmbda_diff_lmm * out["diff_lmm_loss"]

        return out


def train_one_epoch(model_I, model_P, criterion, train_dataloader, optimizer_P, aux_optimizer_P, optimizer_nll_P, train_step, tb_writer=None, clip_max_norm=None):

    model_I.eval()
    model_P.train()
    device = next(model_I.parameters()).device

    for x in tqdm(train_dataloader):
        x = x.to(device).contiguous()
        tau = random.randint(1, 5)
        lastframe = None
        for i in range(1, 8):
            optimizer_P.zero_grad()
            if aux_optimizer_P is not None:
                aux_optimizer_P.zero_grad()
            if optimizer_nll_P is not None:
                optimizer_nll_P.zero_grad()
            if i == 1:
                inp = patchify_to_batch(x[:, 0], 64)
                out = model_I(inp, tau)
                lastframe = unpatchify_from_batch(out["rec_img"].detach(), x[:, i].shape[0], x[:, i].shape[2], x[:, i].shape[3])

            out = model_P(x[:, i], lastframe, tau)

            lastframe = unpatchify_from_batch(out["rec_img"].detach(), x[:, i].shape[0], x[:, i].shape[2], x[:, i].shape[3])

            out_criterion = criterion(out, patchify_to_batch(x[:, i], 64))
            out_criterion["loss"].backward()
            if clip_max_norm:
                nn.utils.clip_grad_norm_(model_P.parameters_wo_cond(), clip_max_norm)
            optimizer_P.step()

            if optimizer_nll_P is not None:
                out_criterion["res_bpp_cond"].backward()
                if clip_max_norm:
                    nn.utils.clip_grad_norm_(model_P.parameters_w_cond(), clip_max_norm)
                optimizer_nll_P.step()

            if aux_optimizer_P is not None:
                aux_loss = model_P.lossy_compressor.aux_loss()
                aux_loss.backward()
                aux_optimizer_P.step()

            if tb_writer is not None and train_step % 50 == 0:
                tb_writer.add_scalar("train/loss", float(out_criterion["loss"].detach()), train_step)
                tb_writer.add_scalar("train/img_bpp", float(out_criterion["img_bpp"].detach()), train_step)
                tb_writer.add_scalar("train/res_bpp", float(out_criterion["res_bpp"].detach()), train_step)
                if "diff_lmm_loss" in out_criterion:
                    tb_writer.add_scalar("train/diff_lmm_loss",
                                         float(out_criterion["diff_lmm_loss"].detach()
                                               if hasattr(out_criterion["diff_lmm_loss"], "detach")
                                               else out_criterion["diff_lmm_loss"]),
                                         train_step)
            train_step += 1
    return train_step


def eval_epoch(model_I, model_P, criterion, eval_dataloader, epoch, save_dir, tb_writer=None):

    model_I.eval()
    model_P.eval()

    device = next(model_I.parameters()).device

    img_bpp_p = 0
    res_bpp_p = 0
    res_bpp_cond_p = 0

    eval_size = 0
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to(device).contiguous()
            tau = random.randint(1, 5)
            eval_size += 1
            lastframe = None
            for i in range(1, 8):
                if i == 1:
                    inp = patchify_to_batch(x[:, 0], 64)
                    out = model_I(inp, tau)
                    lastframe = unpatchify_from_batch(out["rec_img"].detach(), x[:, i].shape[0], x[:, i].shape[2], x[:, i].shape[3])

                out = model_P(x[:, i], lastframe, tau)
                lastframe = unpatchify_from_batch(out["rec_img"].detach(), x[:, i].shape[0], x[:, i].shape[2], x[:, i].shape[3])

                out_criterion = criterion(out, patchify_to_batch(x[:, i], 64))
                img_bpp_p += out_criterion["img_bpp"]
                res_bpp_p += out_criterion["res_bpp"]
                res_bpp_cond_p += out_criterion["res_bpp_cond"]

    img_bpp_p /= eval_size * 7
    res_bpp_p /= eval_size * 7
    res_bpp_cond_p /= eval_size * 7

    torch.save(model_P.state_dict(), f'{save_dir}/p-Ep{epoch + 1}-{img_bpp_p:.3f}-{img_bpp_p + res_bpp_p:.3f}-{img_bpp_p + res_bpp_cond_p:.3f}.pth')


def train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir, resume_ckp=None,
          stage="A", save_dir=SAVE_DIR_DEFAULT, start_epoch_override=None,
          context_type="simple"):
    nll_module_I = NearLosslessCompressor_I(192, 5, "3P").cuda()
    nll_module_P = NearLosslessCompressor_P(192, 5, "3P", context_type=context_type).cuda()

    stage = stage.upper()
    print(f"[stage] running STAGE={stage}")
    print(f"[context] context_type={context_type}")
    if stage == "A":
        rd_criterion = RateDistortion_diff(lmbda=0.0, use_bpp=False, lmbda_diff_lmm=1.0)
    elif stage == "B":
        rd_criterion = RateDistortion_diff(lmbda=0.0, use_bpp=True, lmbda_diff_lmm=0.0)
    elif stage == "C":
        rd_criterion = RateDistortion_diff(lmbda=0.1, use_bpp=True, lmbda_diff_lmm=0.0)
    else:
        raise ValueError(f"Unknown stage={stage}")

    # Load frozen I-frame model
    ckp = torch.load(os.path.join(ckp_dir, "ckp_nll/0001/ckp.tar"))
    # Convert old compressai key format to new format
    state_dict = ckp['model_state_dict']
    new_state_dict = {}
    import re
    for k, v in state_dict.items():
        # _matrix0 -> matrices.0, _bias0 -> biases.0, _factor0 -> factors.0
        new_k = re.sub(r'_matrix(\d+)', r'matrices.\1', k)
        new_k = re.sub(r'_bias(\d+)', r'biases.\1', new_k)
        new_k = re.sub(r'_factor(\d+)', r'factors.\1', new_k)
        new_state_dict[new_k] = v
    nll_module_I.load_state_dict(new_state_dict)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_ckp is not None and os.path.isfile(resume_ckp):
        print(f"Resuming from checkpoint: {resume_ckp}")
        ckp_p = torch.load(resume_ckp, weights_only=False)
        # Backward-compatible loader: accept either a bare state_dict or a wrapper.
        if isinstance(ckp_p, dict) and "model_state_dict" in ckp_p and not any(
            k.startswith(("lossy_compressor.", "mask_conv.", "residual_compressor"))
            for k in ckp_p.keys()
        ):
            ckp_state = ckp_p["model_state_dict"]
        else:
            ckp_state = ckp_p
        nll_module_P.load_state_dict(ckp_state)
        # Parse epoch from filename: p-Ep{epoch}-xxx-xxx-xxx.pth
        basename = os.path.basename(resume_ckp)
        try:
            ep_str = basename.split('-')[1]  # "Ep123"
            start_epoch = int(ep_str.replace('Ep', ''))
            print(f"Resuming from epoch {start_epoch} (parsed from filename)")
        except (IndexError, ValueError):
            start_epoch = 0
            print(f"Could not parse epoch from filename '{basename}', starting at epoch 0")
    if start_epoch_override is not None:
        start_epoch = int(start_epoch_override)
        print(f"start_epoch overridden via --start-epoch: {start_epoch}")

    # === Per-stage freeze / unfreeze ===
    if stage == "A":
        for p in nll_module_P.parameters():
            p.requires_grad_(False)
        for p in nll_module_P.residual_compressor.unet.parameters():
            p.requires_grad_(True)
        nll_module_P.residual_compressor.enable_teacher()
    elif stage == "B":
        for p in nll_module_P.parameters():
            p.requires_grad_(False)
        for p in nll_module_P.residual_compressor.unet.parameters():
            p.requires_grad_(True)
        nll_module_P.residual_compressor.disable_teacher()
    elif stage == "C":
        for p in nll_module_P.parameters():
            p.requires_grad_(True)
        for p in nll_module_P.residual_compressor.teacher.parameters():
            p.requires_grad_(False)
        nll_module_P.residual_compressor.disable_teacher()

    n_trainable = sum(p.numel() for p in nll_module_P.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in nll_module_P.parameters())
    print(f"[stage {stage}] trainable params: {n_trainable:,} / {n_total:,}")

    # === Per-stage optimizers ===
    if stage == "A":
        # Only train residual_compressor.unet; no cond head, no aux update.
        optimizer_P = optim.Adam(
            [p for p in nll_module_P.parameters_wo_cond() if p.requires_grad],
            lr=1e-4,
        )
        aux_optimizer_P = None
        optimizer_nll_P = None
    elif stage == "B":
        # Same partitioning as default; aux still relevant for entropy bottleneck.
        optimizer_P = optim.Adam(
            [p for p in nll_module_P.parameters_wo_cond() if p.requires_grad],
            lr=1e-4,
        )
        aux_params = [
            p for n, p in nll_module_P.lossy_compressor.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        ]
        aux_optimizer_P = optim.Adam(aux_params, lr=1e-3) if aux_params else None
        optimizer_nll_P = None
    else:  # stage == "C"
        optimizer_P, aux_optimizer_P, optimizer_nll_P = configure_optimizers(
            nll_module_P, lr=1e-5, aux_lr=1e-3
        )

    scheduler_P = optim.lr_scheduler.MultiStepLR(
        optimizer_P, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True
    )
    scheduler_nll_P = (
        optim.lr_scheduler.MultiStepLR(
            optimizer_nll_P, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True
        )
        if optimizer_nll_P is not None else None
    )

    # Fast-forward LR schedulers to start_epoch so resumed runs see the
    # correct learning rate (MultiStepLR has no notion of "where we are"
    # other than .step() count).
    for _ in range(start_epoch):
        scheduler_P.step()
        if scheduler_nll_P is not None:
            scheduler_nll_P.step()
    tb_writer = SummaryWriter(log_dir)

    os.makedirs(save_dir, exist_ok=True)

    train_step = 0
    for epoch in range(start_epoch, epochs):
        # Stage A always uses diff_lmm_loss only (no bpp). Stage B/C use bpp from epoch 0.
        # The legacy 200-epoch warm-up only applied to the original MSE+bpp recipe; in
        # Stage A/B/C the criterion is already correctly configured above, so leave it.
        train_step = train_one_epoch(
            nll_module_I, nll_module_P, rd_criterion, train_dataloader,
            optimizer_P, aux_optimizer_P, optimizer_nll_P,
            train_step, tb_writer, clip_max_norm=1.0,
        )
        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] Starting evaluation...")
            eval_epoch(nll_module_I, nll_module_P, rd_criterion, eval_dataloader, epoch, save_dir, tb_writer)

        scheduler_P.step()
        if scheduler_nll_P is not None:
            scheduler_nll_P.step()

    tb_writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pth file to resume training')
    parser.add_argument('--stage', type=str, default="A", choices=["A", "B", "C"],
                        help='Training stage: A=self-distill warmup, B=joint finetune, C=end-to-end')
    parser.add_argument('--epochs', type=int, default=10000, help='Total epoch budget')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR_DEFAULT,
                        help='Directory to save ckpt files for THIS run (Stage A/B/C should use different dirs)')
    parser.add_argument('--start-epoch', type=int, default=None,
                        help='Override start epoch (otherwise parsed from --resume filename p-Ep{N}-...pth)')
    parser.add_argument('--context-type', type=str, default="simple",
                        choices=["simple", "diffusion_fusion"],
                        help='Residual context module type.')
    parser.add_argument('--gpu-id', '--cuda-visible-devices', dest='gpu_id',
                        type=str, default=CUDA_VISIBLE_DEVICES,
                        help='CUDA_VISIBLE_DEVICES value, for example 0 or 0,1.')
    args = parser.parse_args()

    epochs = args.epochs

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckp_dir = ""
    log_dir = "logs4_diff"

    os.makedirs(log_dir, exist_ok=True)
    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    train_dataset = Dataset_seq("../Mosmed/train-full")
    test_dataset = Dataset_test("../Mosmed/test-slice")

    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)
    eval_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=2, pin_memory=True)

    train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir,
          resume_ckp=args.resume, stage=args.stage, save_dir=args.save_dir,
          start_epoch_override=args.start_epoch, context_type=args.context_type)


# ============================================================================
# Pipeline (run from DLPR-Mosmed-nll/, in order):
#
# (0) Prepare Stage-A starting ckp from a baseline (old ResidualCompressor head):
#     python init_diff_lmm_from_baseline.py \
#         --old-ckp ./log4/p-Ep1251-0.005-4.796-5.067.pth \
#         --new-ckp ./log4/p_init_with_teacher.pth
#
# (1) Stage A — self-distillation warmup (unet learns to copy teacher):
#     python train4_diff_sit.py --stage A \
#         --resume ./log4/p_init_with_teacher.pth \
#         --save-dir log4_warmup_stageA --epochs 20000
#
# (2) Stage B — joint finetune with res_bpp loss, teacher off, K=3 DDIM (truncated BPTT):
#     python train4_diff_sit.py --stage B \
#         --resume ./log4_warmup_stageA/p-Ep501-0.005-5.102-5.065.pth \
#         --save-dir log4_warmup_stageB --epochs 20000
#
# (3) Stage C — end-to-end finetune, lr=1e-5:
#     python train4_diff_sit.py --stage C \
#         --resume ./log4_warmup_stageB/p-Ep{}-{}-{}-{}.pth \
#         --save-dir log4_warmup_stageC --epochs 20000
#
# Resume mid-stage: just re-point --resume to the latest p-Ep{N}-...pth in the
# same --save-dir; start_epoch is parsed from the filename and the LR scheduler
# is fast-forwarded to match. Use --start-epoch N if your filename does not
# follow the p-Ep{N}-...pth convention.
# ============================================================================
