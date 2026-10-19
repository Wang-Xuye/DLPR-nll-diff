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


# Rolling training state written next to the weights inside --save-dir. It lets a
# resumed run continue with the same Adam moments / LR / RNG instead of starting
# the optimizers from scratch, and it records which weights file it belongs to so
# it can never be paired with the wrong checkpoint.
TRAIN_STATE_NAME = "last_train_state.pth"
TRAIN_STATE_FORMAT = "static-b-diff-train-state-v1"

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


def resolve_resume_path(resume, save_dir):
    """Resolve the --resume shortcuts.

    "latest" / "auto"   -> highest-epoch p-Ep*.pth inside --save-dir
    a directory path    -> highest-epoch p-Ep*.pth inside that directory
                           (handy for stage transitions, e.g. --resume ./log_static_b_diff_A)
    anything else       -> returned unchanged (a concrete .pth path)
    """
    if resume in ("latest", "auto"):
        search_dir, label = save_dir, resume
    elif resume and os.path.isdir(resume):
        search_dir, label = resume, resume
    else:
        return resume
    if not os.path.isdir(search_dir):
        raise RuntimeError(f"--resume {label}: directory {search_dir} does not exist")
    candidates = [
        name
        for name in os.listdir(search_dir)
        if name.endswith(".pth") and re.match(r"p-Ep(\d+)-", name)
    ]
    if not candidates:
        raise RuntimeError(f"--resume {label}: no p-Ep*.pth found under {search_dir}")

    def epoch_of(name):
        return int(re.match(r"p-Ep(\d+)-", name).group(1))

    # Newest first, but skip unreadable files: torch.save() is not atomic, so a run
    # killed mid-save leaves a truncated *.pth whose name still looks valid. Without
    # this skip, "--resume latest" would become unusable exactly when it is needed.
    ordered = sorted(candidates, key=epoch_of, reverse=True)
    skipped = []
    for name in ordered:
        path = os.path.join(search_dir, name)
        if probe_checkpoint(path):
            if skipped:
                print(f"[warn] skipped unreadable checkpoint(s): {', '.join(skipped)}")
            print(f"[info] --resume {label} -> {path}")
            return path
        skipped.append(name)
    raise RuntimeError(
        f"--resume {label}: none of the {len(candidates)} p-Ep*.pth under {search_dir} "
        f"is a readable checkpoint:\n  " + "\n  ".join(skipped)
    )


def probe_checkpoint(path):
    """Cheap 'is this a readable torch checkpoint' test.

    map_location="meta" parses the archive (so a truncated file fails here) without
    materialising any tensor, which keeps it fast even for a 400 MB state_dict.
    """
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        torch.load(path, map_location="meta", weights_only=False)
        return True
    except Exception:
        return False


def atomic_torch_save(obj, path):
    """torch.save() to a temporary name, then rename.

    torch.save() is not atomic: a run killed while writing leaves a truncated file
    under the final name, which later fails with
    "PytorchStreamReader failed reading zip archive: failed finding central directory".
    """
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def restore_rng(rng_state):
    """Restore python / numpy / torch / cuda RNG states saved by save_train_state."""
    if not rng_state:
        return
    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        cuda_states = rng_state.get("cuda")
        if cuda_states and torch.cuda.is_available():
            # a checkpoint saved with more GPUs than are visible now is truncated
            keep = min(len(cuda_states), max(torch.cuda.device_count(), 1))
            torch.cuda.set_rng_state_all(list(cuda_states)[:keep])
    except Exception as err:  # RNG restore is a convenience, never fatal
        print(f"[warn] could not fully restore RNG state: {err}")


def save_train_state(ckpt_path, completed_epochs, stage, optimizer_p, aux_optimizer_p,
                     optimizer_nll_p, train_step):
    """Write the rolling optimizer/RNG state that pairs with ckpt_path.

    The weights themselves are not duplicated here: ckpt_path stays the anchor and
    "ckpt_name" makes sure a state file can never be applied to another checkpoint.
    "stage" records which objective (Stage A = diffusion only, Stage B = joint) the
    optimizers belong to, so resuming with the wrong --stage can be flagged.
    """
    state = {
        "format": TRAIN_STATE_FORMAT,
        "ckpt_name": os.path.basename(ckpt_path),
        "completed_epochs": completed_epochs,
        "stage": stage,
        "train_step": train_step,
        "optimizer_p": optimizer_p.state_dict(),
        "aux_optimizer_p": aux_optimizer_p.state_dict(),
        "optimizer_nll_p": optimizer_nll_p.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    state_path = os.path.join(os.path.dirname(ckpt_path) or ".", TRAIN_STATE_NAME)
    atomic_torch_save(state, state_path)
    print(
        f"[state] saved {state_path} (pairs with {state['ckpt_name']}, "
        f"stage={stage}, epochs={completed_epochs})"
    )


def find_train_state(ckpt_path, reset_optim):
    """Load the rolling state that belongs to ckpt_path, or None with a reason."""
    if reset_optim:
        print("[state] --reset-optim given: optimizers/RNG restart from scratch")
        return None
    state_path = os.path.join(os.path.dirname(ckpt_path) or ".", TRAIN_STATE_NAME)
    if not os.path.isfile(state_path):
        print(f"[state] no {state_path}: resuming weights only (optimizers restart from scratch)")
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != TRAIN_STATE_FORMAT:
        print(f"[state] {state_path} is not a training state: resuming weights only")
        return None
    if state.get("ckpt_name") != os.path.basename(ckpt_path):
        print(
            f"[state] {state_path} pairs with {state.get('ckpt_name')}, not "
            f"{os.path.basename(ckpt_path)}: resuming weights only"
        )
        return None
    print(
        f"[state] using {state_path} (pairs with {state['ckpt_name']}, "
        f"stage={state.get('stage')}, completed_epochs={state.get('completed_epochs')})"
    )
    return state


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


def resolve_diff_loss_scale(value):
    """Resolve --diff-loss-scale.

    "auto" -> (nii_max_f / 2) ** 2, i.e. the factor that turns an MSE in the
    normalised [-1, 1] domain back into an MSE on the raw [0, 4095] pixels, so
    that --lmbda keeps the meaning it has in train_static_b.py.
    """
    text = str(value).strip().lower()
    if text == "auto":
        return (nii_max_f / 2.0) ** 2
    return float(text)


def warm_start_residual_head(model_p, dataloader, device, n_batches=8):
    """Re-initialise the residual LMM heads to a live operating point.

    A freshly initialised head often predicts a log_sigma far below the actual
    residual scale (the diffusion U-Net features feeding it are large, so the raw
    head output saturates). Every residual likelihood then falls under
    --likelihood-lower-bound, the criterion clamps it flat, and the residual path
    gets exactly zero gradient: res_bpp sits on the ceiling (29.897 bits at 1e-9)
    for the whole run while the LMM stays random. Re-initialising the head as a
    constant predictor (mu = 0, sigma = 1.5 x residual scale) puts it back into the
    live region, from where it can learn.

    The probe uses ground-truth references and therefore slightly under-estimates
    the coding residual, hence the 1.5x margin.
    """
    model = unwrap_model(model_p)
    was_training = model_p.training
    model_p.eval()
    idx0, idx1, idx2 = STATIC_REF_MAP_9[1]
    residuals = []
    with torch.no_grad():
        for i, frames in enumerate(dataloader):
            if i >= n_batches:
                break
            # frames is (B, T, 1, H, W): indexing (not slicing) the GOP axis keeps
            # the reference frames at (B, 1, H, W), which is what the model expects
            frames = frames.to(device)
            # go through the (possibly DataParallel-wrapped) full model so the batch
            # is scattered normally; out["x_hat"] is already patchified
            out = model_p(
                frames[:, idx0],
                frames[:, idx1],
                frames[:, idx2],
                1,
            )
            target = patchify_to_batch(frames[:, idx0], 64)
            residuals.append(((target - out["x_hat"]) / nii_max_f * 2).flatten())
    if was_training:
        model_p.train()
    if not residuals:
        print("[warn] --warm-start-res-head: could not read a batch, skipped")
        return

    res = torch.cat(residuals)
    scale = max(float(res.std()), float(res.abs().mean()))
    sigma = min(max(1.5 * scale, 1e-3), 2.0)
    log_sigma = math.log(sigma)

    # mu = 0, sigma as measured, coeffs = 0, mixture weights = 0 (uniform)
    head = model.residual_compressor.layers[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[5:10].fill_(log_sigma)

    # Conv2d_cond has no conv bias: the per-condition bias lives in bias_cond
    cond_head = model.residual_compressor_cond.conv_cond2
    with torch.no_grad():
        cond_head.weight.zero_()
        bias_cond = getattr(cond_head, "bias_cond", None)
        if bias_cond is not None:
            bias_cond.weight.zero_()
            bias_cond.weight[5:10, :].fill_(log_sigma)

    print(
        f"[init] residual head re-initialised: mu=0, sigma={sigma:.4g} "
        f"(residual |mean|={float(res.abs().mean()):.4g}, "
        f"std={float(res.std()):.4g}, n={res.numel()})"
    )


class StableRateDistortion(nn.Module):
    def __init__(self, lmbda=0.01, likelihood_lower_bound=1e-9, diffusion_only=False,
                 diff_loss_scale=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.likelihood_lower_bound = likelihood_lower_bound
        self.diffusion_only = diffusion_only
        self.diff_loss_scale = diff_loss_scale

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

        # Diagnostics for the "residual head is dead" failure mode: when the
        # predicted sigma is far below the residual scale, every residual
        # likelihood falls under likelihood_lower_bound, the clamp above flattens
        # it, and the head receives exactly zero gradient - res_bpp then sits on
        # the ceiling (29.897 bits at 1e-9) forever. clamp_frac = 1.0 is the
        # unmistakable signature.
        with torch.no_grad():
            min_log = math.log(self.likelihood_lower_bound)
            out["res_clamp_frac"] = (
                (output["res_likelihoods"] <= min_log).float().mean()
            )
            out["res_abs_mean"] = (target - output["x_hat"]).abs().mean()

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
            out["mse_weighted"] = out["mse_loss"]
        else:
            # The diffusion loss lives in the normalised [-1, 1] domain while the
            # plain static-B trainer used MSE in the raw [0, 4095] domain, so the
            # same --lmbda under-weights the distortion by (nii_max_f / 2) ** 2
            # (~4.19e6) here. Left uncompensated the two rate terms dominate by
            # several orders of magnitude and the encoder crushes the latent to
            # ~0 bpp. --diff-loss-scale puts the two recipes on the same scale.
            out["mse_weighted"] = self.lmbda * self.diff_loss_scale * out["mse_loss"]
            out["loss"] = out["mse_weighted"] + out["img_bpp"] + out["res_bpp"]
        return out


def format_criterion_values(out_criterion):
    fields = ("loss", "mse_loss", "mse_weighted", "img_bpp", "res_bpp", "res_bpp_cond",
              "res_clamp_frac", "res_abs_mean")
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
                tb_writer.add_scalar(
                    "train/mse_weighted", float(out_criterion["mse_weighted"].detach()), train_step
                )
                tb_writer.add_scalar("train/img_bpp", float(out_criterion["img_bpp"].detach()), train_step)
                tb_writer.add_scalar("train/res_bpp", float(out_criterion["res_bpp"].detach()), train_step)
                tb_writer.add_scalar(
                    "train/res_bpp_cond", float(out_criterion["res_bpp_cond"].detach()), train_step
                )
            train_step += 1
    return train_step


def eval_epoch(model_i, model_p, criterion, eval_dataloader, epoch, save_dir, tb_writer=None,
               stage="B"):
    model_i.eval()
    model_p.eval()
    device = next(unwrap_model(model_i).parameters()).device

    img_bpp_p = 0.0
    res_bpp_p = 0.0
    res_bpp_cond_p = 0.0
    mse_loss_p = 0.0
    res_clamp_frac_p = 0.0
    res_abs_mean_p = 0.0
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
                res_clamp_frac_p += float(out_criterion["res_clamp_frac"])
                res_abs_mean_p += float(out_criterion["res_abs_mean"])
                frame_count += 1

    if frame_count == 0:
        print(f"[warn] epoch {epoch + 1}: no finite eval frames, checkpoint not saved.")
        return None

    img_bpp_p /= max(frame_count, 1)
    res_bpp_p /= max(frame_count, 1)
    res_bpp_cond_p /= max(frame_count, 1)
    mse_loss_p /= max(frame_count, 1)
    res_clamp_frac_p /= max(frame_count, 1)
    res_abs_mean_p /= max(frame_count, 1)

    if tb_writer is not None:
        tb_writer.add_scalar("eval/img_bpp", img_bpp_p, epoch)
        tb_writer.add_scalar("eval/res_bpp", res_bpp_p, epoch)
        tb_writer.add_scalar("eval/res_bpp_cond", res_bpp_cond_p, epoch)
        # reconstruction quality of the base layer. This is the only meaningful
        # metric while --stage A runs (the residual LMM is still untrained, so the
        # res_bpp numbers in the checkpoint name are meaningless there).
        tb_writer.add_scalar("eval/mse_loss", mse_loss_p, epoch)
        # frac == 1.0 means the residual head is dead (all likelihoods clamped, no
        # gradient); compare res_abs_mean with the head's sigma to see why.
        tb_writer.add_scalar("eval/res_clamp_frac", res_clamp_frac_p, epoch)
        tb_writer.add_scalar("eval/res_abs_mean", res_abs_mean_p, epoch)
        tb_writer.add_scalar("eval/bpp", img_bpp_p + res_bpp_p, epoch)
        tb_writer.add_scalar("eval/bpp_cond", img_bpp_p + res_bpp_cond_p, epoch)
        tb_writer.flush()

    if stage == "A":
        # Stage A has no rate term, so img_bpp/res_bpp are pinned at the
        # likelihood clamp and useless for picking a checkpoint. Put the metric
        # that actually matters (normalised reconstruction MSE) into the name;
        # mse is zero padded to a fixed width so that
        #   ls | grep mse | sort -t- -k3,3 | head -1
        # returns the best checkpoint.
        ckp_name = "p-Ep{}-mse{:08.5f}-imgbpp{:.2f}-resbpp{:.2f}.pth".format(
            epoch + 1,
            mse_loss_p,
            img_bpp_p,
            res_bpp_p,
        )
    else:
        # Stage B: the rate terms are optimised, so the usual
        # p-Ep{N}-{img_bpp}-{img_bpp+res_bpp}-{img_bpp+res_bpp_cond} applies.
        ckp_name = "p-Ep{}-{:.3f}-{:.3f}-{:.3f}.pth".format(
            epoch + 1,
            img_bpp_p,
            img_bpp_p + res_bpp_p,
            img_bpp_p + res_bpp_cond_p,
        )
    ckp_path = os.path.join(save_dir, ckp_name)
    os.makedirs(save_dir, exist_ok=True)
    atomic_torch_save(model_state_dict(model_p), ckp_path)
    print(
        f"[eval] saved {ckp_path} (eval mse_loss={mse_loss_p:.6f}, "
        f"img_bpp={img_bpp_p:.4f}, res_bpp={res_bpp_p:.4f}, "
        f"res_clamp_frac={res_clamp_frac_p:.3f}, res|mean|={res_abs_mean_p:.1f})"
    )
    return ckp_path


def train(args):
    if args.gop_size != 9:
        raise ValueError("Static B-frame ref_map currently supports only --gop-size 9.")

    args.resume = resolve_resume_path(args.resume, args.save_dir)

    device = torch.device("cuda")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model_i = NearLosslessCompressor_I(192, 5, "3P").to(device)
    for param in model_i.parameters():
        param.requires_grad_(False)
    model_p = NearLosslessCompressor_P(192, 5, "3P", embed_dim=args.diff_embed_dim).to(device)

    stage = args.stage.upper()
    diff_loss_scale = resolve_diff_loss_scale(args.diff_loss_scale)
    criterion = StableRateDistortion(
        lmbda=args.lmbda,
        likelihood_lower_bound=args.likelihood_lower_bound,
        diffusion_only=(stage == "A"),
        diff_loss_scale=diff_loss_scale,
    )
    if stage == "A":
        print("[info] stage A: base-layer warm-up, loss = diffusion_loss only (no rate terms)")
    else:
        print(
            f"[info] stage B: joint training, "
            f"loss = {args.lmbda} * {diff_loss_scale:.6g} * diffusion_loss + img_bpp + res_bpp "
            f"(effective distortion weight {args.lmbda * diff_loss_scale:.6g})"
        )
        if args.lmbda * diff_loss_scale < 100:
            print(
                f"[warn] the effective distortion weight lmbda*scale="
                f"{args.lmbda * diff_loss_scale:.3g} is tiny next to the rate terms "
                f"(img_bpp/res_bpp are O(1..30)). The diffusion loss lives in the "
                f"normalised [-1,1] domain, so with --diff-loss-scale 1.0 the rate terms "
                f"dominate by orders of magnitude and the encoder crushes the latent "
                f"(img_bpp -> 0, as in the previous runs). Use "
                f"--diff-loss-scale auto (={(nii_max_f / 2) ** 2:.3g}) to make --lmbda mean "
                f"the same thing it does in train_static_b.py."
            )

    ckp_i = torch.load(args.i_frame_ckp, map_location=device, weights_only=False)
    model_i.load_state_dict(normalize_checkpoint_keys(ckp_i))
    model_i.lossy_compressor.update(force=True)

    resume_state = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"--resume {args.resume}: file not found")
        try:
            ckp_p = torch.load(args.resume, map_location=device, weights_only=False)
        except Exception as err:
            raise RuntimeError(
                f"failed to read {args.resume} "
                f"({os.path.getsize(args.resume) / 2**20:.1f} MiB): it is not a complete "
                f"PyTorch checkpoint. torch.save() is not atomic, so a run killed while "
                f"saving leaves a truncated file behind the correct-looking name. Use an "
                f"older checkpoint, or pass the directory / 'latest' (unreadable files are "
                f"skipped automatically now).\n{err}"
            ) from err
        if isinstance(ckp_p, dict) and ckp_p.get("format") == TRAIN_STATE_FORMAT:
            raise RuntimeError(
                f"{args.resume} is the rolling training state ({TRAIN_STATE_NAME}), "
                f"not a model checkpoint. Point --resume at the matching p-Ep*.pth "
                f"(or use --resume latest); the state is picked up automatically."
            )
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
        resume_state = find_train_state(args.resume, args.reset_optim)
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
    print(
        "[info] train state file="
        + ("disabled (--no-train-state)" if args.no_train_state else TRAIN_STATE_NAME)
    )

    for _ in range(start_epoch):
        scheduler_p.step()
        scheduler_nll_p.step()

    if args.warm_start_res_head:
        if stage == "A":
            print("[info] --warm-start-res-head is ignored in stage A (nothing trains the LMM)")
        else:
            warm_start_residual_head(model_p, train_dataloader, device)

    train_step = 0
    if resume_state is not None:
        # Optimizer moments, LR position (already fast-forwarded above) and RNG are
        # restored so the run really continues where it stopped instead of
        # restarting Adam from zero.
        optimizer_p.load_state_dict(resume_state["optimizer_p"])
        aux_optimizer_p.load_state_dict(resume_state["aux_optimizer_p"])
        optimizer_nll_p.load_state_dict(resume_state["optimizer_nll_p"])
        # load_state_dict() also restores the learning rates stored in the state
        # file. The CLI stays authoritative, so re-apply what the schedulers just
        # computed for start_epoch (this is what makes "--resume ... --lr 1e-5" work).
        for group, lr in zip(optimizer_p.param_groups, scheduler_p.get_last_lr()):
            group["lr"] = lr
        for group, lr in zip(optimizer_nll_p.param_groups, scheduler_nll_p.get_last_lr()):
            group["lr"] = lr
        train_step = int(resume_state.get("train_step", 0))
        restore_rng(resume_state.get("rng"))
        state_stage = str(resume_state.get("stage", "")).upper()
        if state_stage and state_stage != stage:
            print(
                f"[warn] this checkpoint was saved during stage {state_stage} but "
                f"--stage {stage} was requested. That switches the objective mid-run: "
                f"pass --stage {state_stage} to keep training the same stage, or add "
                f"--reset-optim if the change is intentional."
            )
        print(
            f"[state] restored optimizers + RNG, train_step={train_step}, "
            f"lr={optimizer_p.param_groups[0]['lr']:.3e}"
        )

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
            ckp_path = eval_epoch(
                model_i, model_p, criterion, eval_dataloader, epoch, args.save_dir, tb_writer,
                stage=stage,
            )
            if ckp_path is not None and not args.no_train_state:
                save_train_state(
                    ckp_path,
                    epoch + 1,
                    stage,
                    optimizer_p,
                    aux_optimizer_p,
                    optimizer_nll_p,
                    train_step,
                )
        scheduler_p.step()
        scheduler_nll_p.step()

    tb_writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="../Mosmed/train-full")
    parser.add_argument("--eval-path", type=str, default="../Mosmed/test-full")
    parser.add_argument("--i-frame-ckp", type=str, default="ckp_nll/0001/ckp.tar")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to continue from: either a p-Ep{N}-...pth path or "
                             "'latest' to pick the highest-epoch checkpoint inside --save-dir. "
                             "The matching <save-dir>/last_train_state.pth (Adam moments, RNG, "
                             "train_step) is restored automatically when present.")
    parser.add_argument("--reset-optim", action="store_true",
                        help="With --resume: restart the Adam moments / RNG instead of "
                             "restoring them (recommended when switching stage/objective).")
    parser.add_argument("--warm-start-res-head", action="store_true",
                        help="Before joint training, re-initialise the residual LMM heads to "
                             "mu=0 / sigma=1.5x the measured residual scale. Needed when "
                             "res_bpp is pinned on the likelihood clamp (e.g. 29.897): in that "
                             "state the clamp kills the gradient and the head cannot recover "
                             "by itself.")
    parser.add_argument("--no-train-state", action="store_true",
                        help="Do not write the rolling last_train_state.pth. Saves ~1-1.5 GB "
                             "of disk, but a resume can then only restore the weights.")
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
    parser.add_argument("--lmbda", type=float, default=0.03,
                        help="Distortion weight of the diffusion term in the joint stage. The "
                             "diffusion loss is normalised, so combine it with "
                             "--diff-loss-scale auto to get the same meaning as in "
                             "train_static_b.py (otherwise the rate terms dominate by ~4e6).")
    parser.add_argument("--diff-loss-scale", type=str, default="1.0",
                        help="Multiplier applied to the diffusion term inside the joint loss. "
                             "'auto' = (nii_max_f/2)^2 = 4.19e6 rescales the normalised "
                             "[-1,1] MSE back to the raw [0,4095] domain, which is what makes "
                             "--lmbda comparable with train_static_b.py. It never changes the "
                             "logged eval/mse_loss or the checkpoint name.")
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
  域（量级 1e5~1e6），同一个 --lmbda 对失真的压力相差 (4095/2)^2 = 4.19e6 倍。
  直接联合训练时码率项完全主导，编码器会把隐变量压到 ~0（img_bpp≈0），基础层
  不承载信息。所以先 Stage A 只训扩散基础层，再 Stage B 联合训练。
  注意：只分 stage 还不够！Stage B 必须加 --diff-loss-scale auto 把量纲对齐，
  否则码率项依然大 1e6 倍、隐变量照样塌缩（实测 p-Ep1-1.786-... ->
  p-Ep51-0.000-29.897-...）。对齐后 --lmbda 的含义与 train_static_b.py 一致：
    --lmbda 0.03 --diff-loss-scale auto  ≡  --lmbda 1.26e5
  另外 Stage B 应从"未饱和"的 Stage A ckpt 起步（文件名里 imgbpp/resbpp 明显
  小于 23.82/29.90），全饱和的 ckpt 会让残差头没有梯度。

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
  --resume ./log_static_b_diff_A/p-Ep801-mse<xx.xxxxx>-imgbpp<xx.xx>-resbpp<xx.xx>.pth \
  --start-epoch 0 \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_B --log-dir logs_static_b_diff_B \
  --lmbda 0.03 --diff-loss-scale auto --epochs 4000 --eval-interval 50 \
  --lr 1e-4 --aux-lr 1e-3 --gpu-id 0

(3) Stage C（可选）：低 lr 精调
python train_static_b_diff.py --stage B \
  --resume ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
  --start-epoch 0 --lr 1e-5 \
  --save-dir log_static_b_diff_C --log-dir logs_static_b_diff_C --gpu-id 0

(4) 不分阶段一把梭（等于旧行为，只把失真权重 0.03 提到 0.1）
python train_static_b_diff.py --stage B --lmbda 0.03 --diff-loss-scale auto \
  --save-dir log_static_b_diff_v2 --log-dir logs_static_b_diff_v2 --gpu-id 0

(5) 断点续训
# 最省事：自动挑 --save-dir 里 epoch 最大的 p-Ep*.pth
python train_static_b_diff.py --stage B --resume latest \
  --save-dir log_static_b_diff_B --log-dir logs_static_b_diff_B \
  --parallel auto --gpu-id 0,1,2,3 --batch-size 4 --lr 2e-4

# 指定具体 ckpt
python train_static_b_diff.py --stage B \
  --resume ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
  --parallel auto --gpu-id 0,1,2,3 --batch-size 4

# 跨 stage 想"干净重开"优化器（新目标函数，Adam 矩估计从零开始）
python train_static_b_diff.py --stage B --resume latest --reset-optim \
  --start-epoch 0 --save-dir log_static_b_diff_B --log-dir logs_static_b_diff_B

    续训恢复的内容
      * 模型权重（永远是 p-Ep*.pth）
      * 三个优化器的状态（主 / cond / quantiles 的 Adam 矩估计）
      * RNG（python / numpy / torch / cuda）与 train_step
      * epoch 计数（从文件名 p-Ep{N}- 解析）与 LR 调度（自动快进到该 epoch）
    这些优化器/RNG 状态存在 <save-dir>/last_train_state.pth（滚动覆盖，约 1~1.5GB），
    里面记着 ckpt_name，只会配对到它对应的那个 p-Ep*.pth：
      - 从最新 ckpt 续训 -> 自动带上优化器状态（真·断点续训）
      - 从更旧的 ckpt 续训 -> 状态文件不匹配，打印提示后只恢复权重
      - 旧 ckpt（没有状态文件）-> 同上，行为与以前一致
    --reset-optim      不恢复优化器/RNG，只恢复权重
    --no-train-state   不写状态文件（省 1~1.5GB 磁盘，但只能恢复权重）
    --start-epoch N    只覆盖 epoch 计数与 LR 位置，不影响是否恢复优化器状态

    单卡和 DataParallel 存出来的权重都能直接续训：保存时统一用
    unwrap_model(...).state_dict()（不带 module. 前缀），加载时
    normalize_checkpoint_keys() 会自动剥掉 module. / _orig_mod. 前缀和
    {"model_state_dict": ...} 外壳。启动时会打印识别结果，例如
    [info] resume checkpoint: 122 keys with 'module.' (multi-GPU save)
    注意 log_static_b/ 的普通版 ckpt 与扩散版参数集不同，不能在这里续训，
    脚本会直接报错提示。
    ckpt 与状态文件都只在 eval 时写（每 --eval-interval 个 epoch），想缩短崩溃后
    的回退距离就把 --eval-interval 调小。

(6) 多卡：nn.DataParallel（单进程，--parallel auto 即默认）
# DataParallel 是把一个 batch 切到多张卡上，所以 batch_size 必须 >= GPU 数才有
# 加速；每个样本是一整个 clip（9 帧 512x512），显存基本随 batch 线性增长，
# 半分辨率以下的特征图可以忽略，全分辨率那几层才是大头。

# 4 卡 24GB：batch 4 = 每张卡 1 个 clip，单卡峰值与"单卡 batch 1"基本相同
# 先跑这条 2-epoch 冒烟，用 nvidia-smi 确认不 OOM、4 张卡都有占用
python train_static_b_diff.py --stage A --epochs 2 --eval-interval 1 \
  --parallel auto --gpu-id 0,1,2,3 --batch-size 4 --lr 2e-4 \
  --num-workers 8 \
  --save-dir log_static_b_diff_smoke --log-dir logs_static_b_diff_smoke

# Stage A 预热（4 卡）
python train_static_b_diff.py --stage A \
  --parallel auto --gpu-id 0,1,2,3 --batch-size 4 --lr 2e-4 --aux-lr 1e-3 \
  --num-workers 8 \
  --train-path ../Mosmed/train-full --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_A --log-dir logs_static_b_diff_A \
  --epochs 800 --eval-interval 50

# Stage B 联合训练（4 卡，从 Stage A 最新 ckpt 接上）
python train_static_b_diff.py --stage B \
  --resume ./log_static_b_diff_A/p-Ep801-mse<xx.xxxxx>-imgbpp<xx.xx>-resbpp<xx.xx>.pth \
  --start-epoch 0 \
  --parallel auto --gpu-id 0,1,2,3 --batch-size 4 --lr 2e-4 --aux-lr 1e-3 \
  --num-workers 8 \
  --train-path ../Mosmed/train-full --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff_B --log-dir logs_static_b_diff_B \
  --lmbda 0.03 --diff-loss-scale auto --epochs 4000 --eval-interval 50

# 显存不够时的两种退法
#   1) 降到 batch 2 + 只用 2 张卡（每张卡仍是 1 个 clip），lr 相应回到 1e-4
#      python train_static_b_diff.py --stage A --gpu-id 0,1 --batch-size 2 --lr 1e-4 ...
#   2) --diff-embed-dim 32：UNet 通道减半，激活大约降到 1/2~1/4；但这会改变网络
#      结构，必须重新训练，测试时也要带 --embed-dim 32

# 只用某几张卡
CUDA_VISIBLE_DEVICES=2,3 python train_static_b_diff.py --gpu-id 2,3 --batch-size 2 ...

# 单卡：batch_size 只能取 1 时不要开 DataParallel（没有加速，还会切出空分片）
python train_static_b_diff.py --parallel none --gpu-id 0 ...

# 注：--epochs / --milestones 以"数据遍历次数"计。batch 4 的 1 个 epoch 仍然是
#     整个训练集，所以 epochs 不用乘 4；只是每个 epoch 的优化步数变成 1/4，
#     而每步的梯度来自 4 个 clip（更稳），因此 lr 建议从 1e-4 提到 2e-4。

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
  ckpt 只在 eval 时保存（每 --eval-interval 个 epoch），保存是"写 .tmp 再 rename"的
  原子操作，所以进程被杀不会留下截断的 .pth；--resume latest / --resume <目录> 会
  跳过读不出来的 ckpt（torch.save 本身不是原子的，旧 run 里可能残留坏文件）。
  文件名按 stage 区分：
    Stage A: p-Ep{N}-mse{08.5f}-imgbpp{xx.xx}-resbpp{xx.xx}.pth
    Stage B: p-Ep{N}-{img_bpp}-{img_bpp+res_bpp}-{img_bpp+res_bpp_cond}.pth
  Stage A 期间码率项不在 loss 里（残差 LMM 也拿不到梯度），img_bpp/res_bpp 会一路
  撞到 --likelihood-lower-bound 的饱和上限（512x512 时 img<=23.82、res<=29.90），
  所以 Stage A 的名字里放真正能选的指标：归一化域的 eval mse（零填充，字典序=数值序）
    ls log_static_b_diff_A | grep mse | sort -t- -k3,3 -k2.3,2n | head -1   # 最优 ckpt
  Stage B 打开码率项后才按码率命名。
  如果 Stage B 的 ckpt 里 res_bpp 精确等于 29.897（res_bpp_cond 也一样，即文件名
  第 3 个数 == 第 2 个数），说明残差头"死了"：LMM 预测的 sigma 远小于残差幅度，
  所有 likelihood 被 --likelihood-lower-bound 压平、梯度恒为 0，头从随机初始化起
  再也没学过。此时加 --warm-start-res-head 重开，它会把 LMM 头重初始化为
  mu=0 / sigma=1.5x 实测残差尺度；配套看 eval/res_clamp_frac（=1.0 即死）与
  eval/res_abs_mean（残差平均幅度）判断是 sigma 太小还是基础层太差。
  Stage B 的 --lmbda 要配合 --diff-loss-scale auto 使用：loss 里扩散项写成
  lmbda * diff_loss_scale * diffusion_loss，scale=auto 时 = (4095/2)^2 = 4.19e6，
  把归一化域的 MSE 折算回原始像素域，--lmbda 才和 train_static_b.py 同义
  （--lmbda 0.03 --diff-loss-scale auto ≡ --lmbda 1.26e5）。默认 scale=1.0 时脚本
  会在启动时打印 [warn]，因为那种配置下码率项比失真项大 1e6 倍、必然塌缩。
  该因子不影响 eval/mse_loss 与文件名里的 mse，调参方向：总 bpp 偏高就调小
  --lmbda，重建质量差就调大 --lmbda。TensorBoard 里看 train/mse_weighted 与
  train/img_bpp / train/res_bpp 是否同量级，用来判断平衡是否合理。
'''




