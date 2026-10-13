import argparse
import os
import random
import re
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

from nll_model_b_diff import (
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
    """Make a state_dict loadable no matter how it was saved.

    Handles three real-world wrappings:
      * {"model_state_dict": ...}                (older train scripts)
      * "module." prefix                         (nn.DataParallel saves)
      * "_orig_mod." prefix                      (torch.compile saves)
    plus the old compressai entropy-bottleneck parameter names.
    """
    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix(state_dict, prefix="module.")
    state_dict = remove_module_prefix(state_dict, prefix="_orig_mod.")
    return convert_entropy_bottleneck_keys(state_dict)


def checkpoint_origin(checkpoint):
    """Short human-readable description of how a checkpoint was saved."""
    state_dict = extract_state_dict(checkpoint)
    wrapped = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    n_module = sum(1 for k in state_dict if k.startswith("module."))
    n_orig_mod = sum(1 for k in state_dict if k.startswith("_orig_mod."))
    tags = []
    if wrapped:
        tags.append("model_state_dict wrapper")
    if n_module:
        tags.append(f"{n_module} keys with 'module.' (multi-GPU save)")
    if n_orig_mod:
        tags.append(f"{n_orig_mod} keys with '_orig_mod.' (torch.compile save)")
    return ", ".join(tags) if tags else "plain state_dict (single-GPU save)"


def parse_resume_epoch(path):
    if not path:
        return 0
    match = re.search(r"p-Ep(\d+)-", os.path.basename(path))
    if match is None:
        return 0
    return int(match.group(1))
def unwrap_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def model_state_dict(model):
    return unwrap_model(model).state_dict()


def maybe_wrap_data_parallel(model, parallel_mode):
    if parallel_mode == "none":
        print("[info] data parallel disabled")
        return model
    visible_gpus = torch.cuda.device_count()
    if visible_gpus > 1:
        print(f"[info] using DataParallel on {visible_gpus} visible GPUs")
        return nn.DataParallel(model)
    if parallel_mode == "data_parallel":
        print("[warn] --parallel data_parallel requested but fewer than 2 GPUs are visible; using single GPU")
    else:
        print("[info] single GPU training")
    return model


def configure_optimizers(model, lr, aux_lr):
    model = unwrap_model(model)
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
    def __init__(self, lmbda=0.01, likelihood_lower_bound=1e-9, diffusion_only=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.likelihood_lower_bound = likelihood_lower_bound
        self.diffusion_only = diffusion_only

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

        if "diffusion_loss" in output:
            out["mse_loss"] = output["diffusion_loss"].mean()
        else:
            # eval branch: model.eval() reconstructs through DDIM sampling and has
            # no diffusion loss to report, so measure the reconstruction in the
            # same normalised [-1, 1] domain as the training loss, otherwise the
            # train/eval curves would be on scales that differ by ~nii_max_f**2.
            x_hat = torch.clamp(output["x_hat"], min=-0.5, max=nii_max_f + 0.5)
            out["mse_loss"] = self.mse(
                x_hat / nii_max_f * 2 - 1,
                target / nii_max_f * 2 - 1,
            )

        if self.diffusion_only:
            # Stage A optimises the generative base layer only.
            # Training: the model returns the normalised, SNR-weighted diffusion
            # loss. Eval: model.eval() does not produce a diffusion loss at all
            # (it runs DDIM sampling), so fall back to the plain reconstruction
            # MSE - eval only reports metrics, and the bpp terms are meaningless
            # while the residual LMM is still untrained.
            out["loss"] = out["mse_loss"]
        else:
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


def normalize_clip_layout(clip, clip_path=None):
    """Return the clip as (T, H, W).

    The .npy clips are stored as (H, W, T); a clip that already is (T, H, W)
    is returned untouched. The two are told apart by the last axis, which is
    only unambiguous while the spatial dims are square (true for this
    dataset). The flip is reported so that a wrong guess is visible in the
    log instead of silently scrambling the frame axis.
    """
    if clip.ndim != 3:
        raise ValueError(f"expected a 3-D clip, got shape {clip.shape} ({clip_path})")
    if clip.shape[-1] != clip.shape[-2]:
        print(f"[layout] {clip_path}: reading {clip.shape} as (H, W, T)")
        clip = clip.transpose(2, 0, 1)
    return clip


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
                clip = normalize_clip_layout(clip, full_path)
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
    with torch.no_grad():
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
    train_step,
    tb_writer=None,
    clip_max_norm=None,
):
    model_i.eval()
    model_p.train()
    device = next(unwrap_model(model_i).parameters()).device

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
                nn.utils.clip_grad_norm_(unwrap_model(model_p).parameters_wo_cond(), clip_max_norm)
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

            if getattr(criterion, "diffusion_only", False):
                # Stage A: the residual LMM / cond head is not part of the
                # objective, so skip its backward (no gradient, no step).
                optimizer_nll_p.zero_grad()
            else:
                out_criterion["res_bpp_cond"].backward()
                if clip_max_norm:
                    nn.utils.clip_grad_norm_(unwrap_model(model_p).parameters_w_cond(), clip_max_norm)
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

            aux_loss = unwrap_model(model_p).lossy_compressor.aux_loss()
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

            if tb_writer is not None and train_step % 50 == 0:
                tb_writer.add_scalar("train/loss", float(out_criterion["loss"].detach()), train_step)
                tb_writer.add_scalar("train/mse_loss", float(out_criterion["mse_loss"].detach()), train_step)
                tb_writer.add_scalar("train/img_bpp", float(out_criterion["img_bpp"].detach()), train_step)
                tb_writer.add_scalar("train/res_bpp", float(out_criterion["res_bpp"].detach()), train_step)
                tb_writer.add_scalar(
                    "train/res_bpp_cond", float(out_criterion["res_bpp_cond"].detach()), train_step
                )
            train_step += 1
    return train_step


def eval_epoch(model_i, model_p, criterion, eval_dataloader, epoch, save_dir, tb_writer=None):
    model_i.eval()
    model_p.eval()
    device = next(unwrap_model(model_i).parameters()).device

    img_bpp_p = 0.0
    res_bpp_p = 0.0
    res_bpp_cond_p = 0.0
    mse_loss_p = 0.0
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
                mse_loss_p += float(out_criterion["mse_loss"].detach())
                frame_count += 1

    if frame_count == 0:
        print(f"[warn] epoch {epoch + 1}: no finite eval frames, checkpoint not saved.")
        return

    img_bpp_p /= max(frame_count, 1)
    res_bpp_p /= max(frame_count, 1)
    res_bpp_cond_p /= max(frame_count, 1)
    mse_loss_p /= max(frame_count, 1)

    if tb_writer is not None:
        tb_writer.add_scalar("eval/img_bpp", img_bpp_p, epoch)
        tb_writer.add_scalar("eval/res_bpp", res_bpp_p, epoch)
        tb_writer.add_scalar("eval/res_bpp_cond", res_bpp_cond_p, epoch)
        # reconstruction quality of the base layer. This is the only meaningful
        # metric while --stage A runs (the residual LMM is still untrained, so the
        # res_bpp numbers in the checkpoint name are meaningless there).
        tb_writer.add_scalar("eval/mse_loss", mse_loss_p, epoch)
        tb_writer.add_scalar("eval/bpp", img_bpp_p + res_bpp_p, epoch)
        tb_writer.add_scalar("eval/bpp_cond", img_bpp_p + res_bpp_cond_p, epoch)
        tb_writer.flush()

    ckp_path = os.path.join(
        save_dir,
        "p-Ep{}-{:.3f}-{:.3f}-{:.3f}.pth".format(
            epoch + 1,
            img_bpp_p,
            img_bpp_p + res_bpp_p,
            img_bpp_p + res_bpp_cond_p,
        ),
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_state_dict(model_p), ckp_path)
    print(f"[eval] saved {ckp_path} (eval mse_loss={mse_loss_p:.6f}, img_bpp={img_bpp_p:.4f})")


def train(args):
    if args.gop_size != 9:
        raise ValueError("Static B-frame ref_map currently supports only --gop-size 9.")

    device = torch.device("cuda")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model_i = NearLosslessCompressor_I(192, 5, "3P").to(device)
    for param in model_i.parameters():
        param.requires_grad_(False)
    model_p = NearLosslessCompressor_P(192, 5, "3P", embed_dim=args.diff_embed_dim).to(device)

    stage = args.stage.upper()
    criterion = StableRateDistortion(
        lmbda=args.lmbda,
        likelihood_lower_bound=args.likelihood_lower_bound,
        diffusion_only=(stage == "A"),
    )
    if stage == "A":
        print("[info] stage A: base-layer warm-up, loss = diffusion_loss only (no rate terms)")
    else:
        print(
            f"[info] stage B: joint training, "
            f"loss = {args.lmbda} * diffusion_loss + img_bpp + res_bpp"
        )

    ckp_i = torch.load(args.i_frame_ckp, map_location=device, weights_only=False)
    model_i.load_state_dict(normalize_checkpoint_keys(ckp_i))
    model_i.lossy_compressor.update(force=True)

    if args.resume:
        ckp_p = torch.load(args.resume, map_location=device, weights_only=False)
        # Single-GPU and DataParallel checkpoints both resume from here: the
        # {"model_state_dict": ...} wrapper and the "module." / "_orig_mod."
        # key prefixes are stripped automatically by normalize_checkpoint_keys.
        print(f"[info] resume checkpoint: {checkpoint_origin(ckp_p)}")
        try:
            model_p.load_state_dict(normalize_checkpoint_keys(ckp_p))
        except RuntimeError as err:
            raise RuntimeError(
                f"failed to load {args.resume} into NearLosslessCompressor_P. A "
                f"checkpoint trained without the diffusion lossy compressor (e.g. "
                f"log_static_b/*.pth) has a different parameter set and cannot be "
                f"resumed here; only log_static_b_diff-style checkpoints fit.\n{err}"
            ) from err
    model_p.lossy_compressor.update(force=True)

    start_epoch = parse_resume_epoch(args.resume)
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    if args.resume:
        print(f"[info] resume={args.resume}")
        print(f"[info] start_epoch={start_epoch}")

    model_i = maybe_wrap_data_parallel(model_i, args.parallel)
    model_p = maybe_wrap_data_parallel(model_p, args.parallel)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus > 1 and args.batch_size < visible_gpus:
        print(
            f"[warn] batch_size={args.batch_size} < visible GPUs={visible_gpus}: "
            f"DataParallel cannot split such a batch, so it brings no speedup and "
            f"may feed empty shards. Use a single GPU (--gpu-id 0 --parallel none) "
            f"or raise --batch-size to at least {visible_gpus}."
        )

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
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
    )
    eval_dataset = DatasetSeq(args.eval_path, gop_size=args.gop_size, random_crop=False)
    eval_dataloader = data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
    )

    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"[info] visible GPU count={torch.cuda.device_count()}")
    print(f"[info] parallel={args.parallel}")
    print("[info] lossy_compressor=diff_sit_static_b")
    print(f"[info] stage={stage}")
    print(f"[info] diff_embed_dim={args.diff_embed_dim}, lmbda={args.lmbda}")
    print(f"[info] static ref_map={STATIC_REF_MAP_9}")
    print(f"[info] train batches per epoch={len(train_dataloader)}")

    for _ in range(start_epoch):
        scheduler_p.step()
        scheduler_nll_p.step()

    train_step = 0
    for epoch in range(start_epoch, args.epochs):
        train_step = train_one_epoch(
            model_i,
            model_p,
            criterion,
            train_dataloader,
            optimizer_p,
            aux_optimizer_p,
            optimizer_nll_p,
            train_step,
            tb_writer=tb_writer,
            clip_max_norm=args.clip_max_norm,
        )
        if epoch % args.eval_interval == 0:
            eval_epoch(model_i, model_p, criterion, eval_dataloader, epoch, args.save_dir, tb_writer)
        scheduler_p.step()
        scheduler_nll_p.step()

    tb_writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="../Mosmed/train-full")
    parser.add_argument("--eval-path", type=str, default="../Mosmed/test-full")
    parser.add_argument("--i-frame-ckp", type=str, default="ckp_nll/0001/ckp.tar")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stage", type=str, default="B", choices=["A", "B"],
                        help="A = generative base-layer warm-up (diffusion loss only, no rate "
                             "terms); B = joint training (lmbda * diffusion_loss + img_bpp + "
                             "res_bpp). Fine-tune a stage-B checkpoint by resuming it with "
                             "--stage B --lr 1e-5.")
    parser.add_argument("--start-epoch", type=int, default=None,
                        help="Override start epoch. Otherwise parsed from --resume p-Ep{N}-...pth.")
    parser.add_argument("--save-dir", type=str, default="log_static_b_diff")
    parser.add_argument("--log-dir", type=str, default="logs_static_b_diff")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gop-size", type=int, default=9)
    parser.add_argument("--lmbda", type=float, default=0.03)
    parser.add_argument("--diff-embed-dim", type=int, default=64,
                        help="Embedding width of the diffusion lossy compressor. Keep 64 for "
                             "checkpoints trained with the default; use 32 to match the P-frame "
                             "diff-sit pipeline / save memory.")
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
    parser.add_argument("--parallel", choices=["auto", "data_parallel", "none"], default="auto",
                        help="Single-process nn.DataParallel mode. 'auto' wraps the model when "
                             "more than one GPU is visible; DataParallel only helps when "
                             "--batch-size >= number of visible GPUs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


r'''
运行指令（工作目录：DLPR-Mosmed-nll/）
====================================================================

路径约定
  ../Mosmed/train-full, ../Mosmed/test-full   训练 / 评估用的 .npy clip 目录
  ckp_nll/0001/ckp.tar                        冻结的 I 帧模型（默认值）
  本脚本产出的 p-Ep{N}-...pth 中：只有 lossy（基础层）压缩器换成了扩散模型，
  残差 LMM 与普通静态 B 帧完全一致，因此两种码率可直接对比。

为什么分两个 stage
  扩散损失在归一化 [-1,1] 域（量级 0.01~1），而普通静态 B 帧的 MSE 在 [0,4095]
  域（量级 1e5~1e6），同一个 --lmbda 对失真的压力相差约 1e6 倍。直接联合训练时
  码率项完全主导，编码器会把隐变量压到 ~0（img_bpp≈0），基础层不承载信息。
  所以先 Stage A 只训扩散基础层，再 Stage B 联合训练。

(0) 冒烟：先确认能跑通（2 epoch，每 epoch 都存 ckpt）
python train_static_b_diff.py --stage A \
  --epochs 2 --eval-interval 1 \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_smoke --log-dir logs_static_b_diff_smoke \
  --gpu-id 0

(1) Stage A：生成式基础层预热（loss = diffusion_loss，不含 img_bpp/res_bpp）
python train_static_b_diff.py --stage A \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_A --log-dir logs_static_b_diff_A \
  --epochs 800 --eval-interval 50 \
  --lr 1e-4 --aux-lr 1e-3 --batch-size 1 --num-workers 8 \
  --gpu-id 0

(2) Stage B：联合训练（loss = lmbda*diffusion_loss + img_bpp + res_bpp）
    --start-epoch 0 = 只加载权重、把 LR 调度重置到满值（否则按文件名快进）
python train_static_b_diff.py --stage B \
  --resume ./log_static_b_diff_A/p-Ep801-<img>-<img+res>-<img+res_cond>.pth \
  --start-epoch 0 \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_B --log-dir logs_static_b_diff_B \
  --lmbda 0.1 --epochs 4000 --eval-interval 50 \
  --lr 1e-4 --aux-lr 1e-3 --gpu-id 0

(3) Stage C（可选）：低 lr 精调
python train_static_b_diff.py --stage B \
  --resume ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
  --start-epoch 0 --lr 1e-5 \
  --save-dir log_static_b_diff_C --log-dir logs_static_b_diff_C --gpu-id 0

(4) 不分阶段一把梭（等于旧行为，只把失真权重 0.03 提到 0.1）
python train_static_b_diff.py --stage B --lmbda 0.1 \
  --save-dir log_static_b_diff_v2 --log-dir logs_static_b_diff_v2 --gpu-id 0

(5) 断点续训（指向同一 save-dir 里最新的 p-Ep{N}-...pth）
python train_static_b_diff.py --stage B \
  --resume ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
  --gpu-id 0

    单卡和 DataParallel 存出来的 ckpt 都能直接续训：保存时统一用
    unwrap_model(...).state_dict()（不带 module. 前缀），加载时
    normalize_checkpoint_keys() 会自动剥掉 module. / _orig_mod. 前缀和
    {"model_state_dict": ...} 外壳。启动时会打印识别结果，例如
    [info] resume checkpoint: 122 keys with 'module.' (multi-GPU save)
    注意 log_static_b/ 的普通版 ckpt 与扩散版参数集不同，不能在这里续训，
    脚本会直接报错提示。

(6) 多卡：nn.DataParallel（单进程，--parallel auto 即默认）
# DataParallel 是把一个 batch 切到多张卡上，所以只有当 batch_size >= GPU 数时
# 才有加速效果。本任务每个样本是一整个 clip（9 帧），batch_size 通常是 1，
# 此时 DataParallel 不会更快，直接单卡跑即可。
python train_static_b_diff.py --stage B --parallel auto --gpu-id 0,1 \
  --batch-size 2 \
  --train-path ../Mosmed/train-full --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_B --log-dir logs_static_b_diff_B \
  --lmbda 0.1 --epochs 4000 --eval-interval 50 \
  --lr 1e-4 --aux-lr 1e-3 --num-workers 8

# 只用某几张卡
CUDA_VISIBLE_DEVICES=2,3 python train_static_b_diff.py --gpu-id 2,3 ...

# 禁用 DataParallel（推荐给 batch_size=1 的 clip 级训练）
python train_static_b_diff.py --parallel none --gpu-id 0 ...

(7) 监控：重点看 train/mse_loss（扩散质量）与 train/img_bpp（基础层码率）
    Stage A 期间看 eval/mse_loss（eval 分支用 DDIM 重建，归一化到与训练同一量纲），
    此时 res_bpp 还没意义。
tensorboard --logdir logs_static_b_diff_B

(8) 训练完成后测试（完整命令见 test_static_b_diff.py 结尾的三引号块）
python test_static_b_diff.py \
  --p-frame-ckp ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
  --i-frame-ckp ckp_nll/0001/ckp.tar --data-path ../Mosmed/test-full \
  --tau 0 --steps 5 --embed-dim 64 --gpu-id 0 \
  --log-file log-static-b-diff-tau-0.txt

多卡注意事项
  * --parallel auto / data_parallel 走的是单进程 nn.DataParallel：模型会复制到
    每张卡，输入 batch 被切分。batch_size < GPU 数时它会切出空分片，没有加速
    效果（脚本会打印提示），这两种情况请用 --parallel none 单卡跑。
  * --gpu-id 里可以写多张卡（如 0,1），DataParallel 会用到全部可见卡；也可以
    直接用 CUDA_VISIBLE_DEVICES 控制。
  * ckpt 统一保存为不带 module. 前缀的 state_dict，所以单卡训练与 DataParallel
    训练之间可以互相续训 / 互相测试。

注意
  --diff-embed-dim 默认 64，必须与测试脚本的 --embed-dim 一致；想和 P 帧
  diff-sit 链路（embed_dim=32）对齐或省显存时用 32。
  务必换新的 --save-dir：脚本没有 next_available_path，会与旧 run 的
  p-Ep{N}-...pth 撞名覆盖。
  ckpt 只在 eval 时保存（每 --eval-interval 个 epoch）；Stage A 的 ckpt 文件名里
  res_bpp 无意义（此时 LMM 未训练），只看 img_bpp，别按它挑 ckpt。
'''




