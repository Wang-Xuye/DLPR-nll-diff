import argparse
import os
import random
import sys


def read_cuda_visible_devices(default="0"):
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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nll_model_b import (
    NearLosslessCompressor_I,
    NearLosslessCompressor_P,
    patchify_to_batch,
    unpatchify_from_batch,
)
from global_nii_max import nii_max_f


STATIC_REF_MAP_9 = [
    [0, 0, 0],
    [8, 0, 0],
    [4, 0, 8],
    [2, 0, 4],
    [1, 0, 2],
    [3, 2, 4],
    [6, 4, 8],
    [5, 4, 6],
    [7, 6, 8],
]


def replicate_pad(x, p):
    height = x.shape[-2]
    width = x.shape[-1]
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    if padding_right == 0 and padding_bottom == 0:
        return x
    return np.pad(x, [(0, 0), (0, padding_bottom), (0, padding_right)], mode="edge")


def remove_module_prefix(state_dict, prefix="module."):
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def convert_entropy_bottleneck_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(
            ".entropy_bottleneck._matrix",
            ".entropy_bottleneck.matrices.",
        )
        new_key = new_key.replace(
            ".entropy_bottleneck._bias",
            ".entropy_bottleneck.biases.",
        )
        new_key = new_key.replace(
            ".entropy_bottleneck._factor",
            ".entropy_bottleneck.factors.",
        )
        new_state_dict[new_key] = value
    return new_state_dict


def normalize_checkpoint_keys(checkpoint):
    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix(state_dict, prefix="module.")
    return convert_entropy_bottleneck_keys(state_dict)


def configure_optimizers(model, lr, aux_lr):
    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles")
        and not n.startswith("residual_compressor_cond.")
        and not n.startswith("fusion_cond.")
        and p.requires_grad
    }
    parameters_nll = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles")
        and p.requires_grad
        and (n.startswith("residual_compressor_cond.") or n.startswith("fusion_cond."))
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

    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=lr)
    optimizer_nll = optim.Adam((params_dict[n] for n in sorted(parameters_nll)), lr=lr)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=aux_lr)
    return optimizer, aux_optimizer, optimizer_nll


def tensor_is_finite(value):
    return torch.isfinite(value.detach()).all().item()


def optimizer_grads_finite(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and not torch.isfinite(param.grad).all().item():
                return False
    return True


class StableRateDistortion(nn.Module):
    def __init__(self, lmbda=0.01, likelihood_lower_bound=1e-9):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.likelihood_lower_bound = likelihood_lower_bound

    def safe_likelihood(self, likelihoods):
        return torch.nan_to_num(
            likelihoods,
            nan=self.likelihood_lower_bound,
            posinf=1.0,
            neginf=self.likelihood_lower_bound,
        ).clamp_min(self.likelihood_lower_bound)

    def safe_log_likelihood(self, log_likelihoods):
        min_log = math.log(self.likelihood_lower_bound)
        return torch.nan_to_num(
            log_likelihoods,
            nan=min_log,
            posinf=0.0,
            neginf=min_log,
        ).clamp(min=min_log, max=0.0)

    def forward(self, output, target):
        n, _, h, w = target.size()
        num_pixels = n * h * w
        out = {}

        out["img_bpp"] = sum(
            torch.log(self.safe_likelihood(likelihoods)).sum()
            / (-math.log(2) * num_pixels)
            for likelihoods in output["img_likelihoods"].values()
        )
        out["res_bpp"] = (
            self.safe_log_likelihood(output["res_likelihoods"]).sum()
            / (-math.log(2) * num_pixels)
        )
        out["res_bpp_cond"] = (
            self.safe_log_likelihood(output["res_likelihoods_cond"]).sum()
            / (-math.log(2) * num_pixels)
        )

        x_hat = torch.clamp(output["x_hat"], min=-0.5, max=nii_max_f + 0.5)
        out["mse_loss"] = self.mse(x_hat, target)
        out["loss"] = self.lmbda * out["mse_loss"] + out["img_bpp"] + out["res_bpp"]
        return out


def format_criterion_values(out_criterion):
    fields = ("loss", "img_bpp", "res_bpp", "res_bpp_cond", "mse_loss")
    values = []
    for field in fields:
        value = out_criterion.get(field)
        if value is None:
            continue
        values.append(f"{field}={float(value.detach().cpu()):.6g}")
    return ", ".join(values)


class DatasetSeq:
    def __init__(self, root_dir, gop_size=9, random_crop=True):
        self.root_dir = root_dir
        self.gop_size = gop_size
        self.random_crop = random_crop
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in sorted(files):
                full_path = os.path.join(root, filename)
                clip = np.load(full_path).astype(np.float32)
                if clip.ndim == 3 and clip.shape[-1] != clip.shape[-2]:
                    clip = clip.transpose(2, 0, 1)
                clip = replicate_pad(clip, 128)
                if clip.shape[0] >= self.gop_size:
                    self.clips.append(clip)
        if not self.clips:
            raise RuntimeError(f"No clips with at least {self.gop_size} frames under {root_dir}.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        t = clip.shape[0]
        if self.random_crop:
            start_idx = np.random.randint(0, t - self.gop_size + 1)
        else:
            start_idx = 0
        clip = clip[start_idx:start_idx + self.gop_size]
        clip = np.expand_dims(clip, axis=1)
        if self.random_crop and random.random() < 0.5:
            clip = clip[:, :, :, ::-1]
        if self.random_crop and random.random() < 0.5:
            clip = clip[:, :, ::-1, :]
        return np.ascontiguousarray(clip, dtype=np.float32)


def encode_i_frame(model_i, frames, tau):
    inp = patchify_to_batch(frames[:, 0], 64)
    out = model_i(inp, tau)
    rec = unpatchify_from_batch(
        out["rec_img"].detach(),
        frames[:, 0].shape[0],
        frames[:, 0].shape[2],
        frames[:, 0].shape[3],
    )
    return out, rec


def train_one_epoch(
    model_i,
    model_p,
    criterion,
    train_dataloader,
    optimizer_p,
    aux_optimizer_p,
    optimizer_nll_p,
    clip_max_norm=None,
):
    model_i.eval()
    model_p.train()
    device = next(model_i.parameters()).device

    for frames in tqdm(train_dataloader, desc="train"):
        frames = frames.to(device).contiguous()
        tau = random.randint(1, 5)
        batch, gop_size, _, height, width = frames.shape

        ref_buffer = torch.zeros((batch, gop_size, height, width), device=device)
        _, rec_i = encode_i_frame(model_i, frames, tau)
        ref_buffer[:, 0:1] = rec_i

        for step in range(1, gop_size):
            optimizer_p.zero_grad()
            aux_optimizer_p.zero_grad()
            optimizer_nll_p.zero_grad()

            idx0, idx1, idx2 = STATIC_REF_MAP_9[step]
            out = model_p(
                frames[:, idx0],
                ref_buffer[:, idx1:idx1 + 1],
                ref_buffer[:, idx2:idx2 + 1],
                tau,
            )

            target = patchify_to_batch(frames[:, idx0], 64)
            out_criterion = criterion(out, target)
            if (
                not tensor_is_finite(out_criterion["loss"])
                or not tensor_is_finite(out_criterion["res_bpp_cond"])
                or not tensor_is_finite(out["rec_img"])
            ):
                print(
                    f"[warn] skip non-finite training step: "
                    f"tau={tau}, step={step}, target={idx0}, "
                    f"{format_criterion_values(out_criterion)}, "
                    f"rec_finite={tensor_is_finite(out['rec_img'])}"
                )
                optimizer_p.zero_grad()
                aux_optimizer_p.zero_grad()
                optimizer_nll_p.zero_grad()
                break

            rec = unpatchify_from_batch(
                out["rec_img"].detach(),
                frames[:, idx0].shape[0],
                frames[:, idx0].shape[2],
                frames[:, idx0].shape[3],
            )
            ref_buffer[:, idx0:idx0 + 1] = rec

            out_criterion["loss"].backward()
            if clip_max_norm:
                nn.utils.clip_grad_norm_(model_p.parameters_wo_cond(), clip_max_norm)
            if not optimizer_grads_finite(optimizer_p):
                print(
                    f"[warn] skip non-finite main gradients: "
                    f"tau={tau}, step={step}, target={idx0}"
                )
                optimizer_p.zero_grad()
                aux_optimizer_p.zero_grad()
                optimizer_nll_p.zero_grad()
                break
            optimizer_p.step()

            out_criterion["res_bpp_cond"].backward()
            if clip_max_norm:
                nn.utils.clip_grad_norm_(model_p.parameters_w_cond(), clip_max_norm)
            if not optimizer_grads_finite(optimizer_nll_p):
                print(
                    f"[warn] skip non-finite residual gradients: "
                    f"tau={tau}, step={step}, target={idx0}"
                )
                optimizer_p.zero_grad()
                aux_optimizer_p.zero_grad()
                optimizer_nll_p.zero_grad()
                break
            optimizer_nll_p.step()

            aux_loss = model_p.lossy_compressor.aux_loss()
            if tensor_is_finite(aux_loss):
                aux_loss.backward()
                if optimizer_grads_finite(aux_optimizer_p):
                    aux_optimizer_p.step()
                else:
                    print(
                        f"[warn] skip non-finite aux gradients: "
                        f"tau={tau}, step={step}, target={idx0}"
                    )
                    aux_optimizer_p.zero_grad()
            else:
                print(
                    f"[warn] skip non-finite aux loss: "
                    f"tau={tau}, step={step}, target={idx0}"
                )
                aux_optimizer_p.zero_grad()


def eval_epoch(model_i, model_p, criterion, eval_dataloader, epoch, save_dir):
    model_i.eval()
    model_p.eval()
    device = next(model_i.parameters()).device

    img_bpp_p = 0.0
    res_bpp_p = 0.0
    res_bpp_cond_p = 0.0
    frame_count = 0

    with torch.no_grad():
        for frames in tqdm(eval_dataloader, desc="eval"):
            frames = frames.to(device).contiguous()
            tau = random.randint(1, 5)
            batch, gop_size, _, height, width = frames.shape

            ref_buffer = torch.zeros((batch, gop_size, height, width), device=device)
            _, rec_i = encode_i_frame(model_i, frames, tau)
            ref_buffer[:, 0:1] = rec_i

            for step in range(1, gop_size):
                idx0, idx1, idx2 = STATIC_REF_MAP_9[step]
                out = model_p(
                    frames[:, idx0],
                    ref_buffer[:, idx1:idx1 + 1],
                    ref_buffer[:, idx2:idx2 + 1],
                    tau,
                )
                target = patchify_to_batch(frames[:, idx0], 64)
                out_criterion = criterion(out, target)
                if (
                    not tensor_is_finite(out_criterion["img_bpp"])
                    or not tensor_is_finite(out_criterion["res_bpp"])
                    or not tensor_is_finite(out_criterion["res_bpp_cond"])
                    or not tensor_is_finite(out["rec_img"])
                ):
                    print(
                        f"[warn] skip non-finite eval step: "
                        f"tau={tau}, step={step}, target={idx0}, "
                        f"{format_criterion_values(out_criterion)}, "
                        f"rec_finite={tensor_is_finite(out['rec_img'])}"
                    )
                    break

                rec = unpatchify_from_batch(
                    out["rec_img"].detach(),
                    frames[:, idx0].shape[0],
                    frames[:, idx0].shape[2],
                    frames[:, idx0].shape[3],
                )
                ref_buffer[:, idx0:idx0 + 1] = rec

                img_bpp_p += float(out_criterion["img_bpp"].detach())
                res_bpp_p += float(out_criterion["res_bpp"].detach())
                res_bpp_cond_p += float(out_criterion["res_bpp_cond"].detach())
                frame_count += 1

    if frame_count == 0:
        print(f"[warn] epoch {epoch + 1}: no finite eval frames, checkpoint not saved.")
        return

    img_bpp_p /= max(frame_count, 1)
    res_bpp_p /= max(frame_count, 1)
    res_bpp_cond_p /= max(frame_count, 1)

    ckp_path = os.path.join(
        save_dir,
        "p-Ep{}-{:.3f}-{:.3f}-{:.3f}.pth".format(
            epoch + 1,
            img_bpp_p,
            img_bpp_p + res_bpp_p,
            img_bpp_p + res_bpp_cond_p,
        ),
    )
    torch.save(model_p.state_dict(), ckp_path)
    print(f"[eval] saved {ckp_path}")


def train(args):
    if args.gop_size != 9:
        raise ValueError("Static B-frame ref_map currently supports only --gop-size 9.")

    device = torch.device("cuda")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model_i = NearLosslessCompressor_I(192, 5, "3P").to(device)
    model_p = NearLosslessCompressor_P(192, 5, "3P").to(device)
    criterion = StableRateDistortion(
        lmbda=args.lmbda,
        likelihood_lower_bound=args.likelihood_lower_bound,
    )

    ckp_i = torch.load(args.i_frame_ckp, map_location=device, weights_only=False)
    model_i.load_state_dict(normalize_checkpoint_keys(ckp_i))
    model_i.lossy_compressor.update(force=True)

    if args.resume:
        ckp_p = torch.load(args.resume, map_location=device, weights_only=False)
        model_p.load_state_dict(normalize_checkpoint_keys(ckp_p))
    model_p.lossy_compressor.update(force=True)

    optimizer_p, aux_optimizer_p, optimizer_nll_p = configure_optimizers(
        model_p, lr=args.lr, aux_lr=args.aux_lr
    )
    scheduler_p = optim.lr_scheduler.MultiStepLR(
        optimizer_p, milestones=args.milestones, gamma=args.gamma
    )
    scheduler_nll_p = optim.lr_scheduler.MultiStepLR(
        optimizer_nll_p, milestones=args.milestones, gamma=args.gamma
    )
    tb_writer = SummaryWriter(args.log_dir)

    train_dataset = DatasetSeq(args.train_path, gop_size=args.gop_size, random_crop=True)
    eval_dataset = DatasetSeq(args.eval_path, gop_size=args.gop_size, random_crop=False)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
    )
    eval_dataloader = data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
    )

    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"[info] static ref_map={STATIC_REF_MAP_9}")

    for epoch in range(args.epochs):
        train_one_epoch(
            model_i,
            model_p,
            criterion,
            train_dataloader,
            optimizer_p,
            aux_optimizer_p,
            optimizer_nll_p,
            clip_max_norm=args.clip_max_norm,
        )
        if epoch % args.eval_interval == 0:
            eval_epoch(model_i, model_p, criterion, eval_dataloader, epoch, args.save_dir)
        scheduler_p.step()
        scheduler_nll_p.step()

    tb_writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="../Mosmed/train-full")
    parser.add_argument("--eval-path", type=str, default="../Mosmed/test-full")
    parser.add_argument("--i-frame-ckp", type=str, default="ckp_nll/0001/ckp.tar")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="log_static_b")
    parser.add_argument("--log-dir", type=str, default="logs_static_b")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gop-size", type=int, default=9)
    parser.add_argument("--lmbda", type=float, default=0.03)
    parser.add_argument("--likelihood-lower-bound", type=float, default=1e-9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--milestones", nargs="+", type=int,
                        default=[350, 390, 430, 470, 510, 550, 590])
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--clip-max-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", "--cuda-visible-devices", dest="gpu_id",
                        type=str, default=CUDA_VISIBLE_DEVICES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
