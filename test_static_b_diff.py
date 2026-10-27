import argparse
import os
import re
import sys
import time


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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchac

from global_nii_max import nii_max, nii_max_f
from nll_model_b_diff import (
    NearLosslessCompressor_I,
    NearLosslessCompressor_P,
    patchify_to_batch,
    unpatchify_from_batch,
)


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


enc_lossy_time = 0.0
enc_res_time = 0.0
dec_lossy_time = 0.0
dec_res_time = 0.0


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


class Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in sorted(files):
                if filename.endswith(".npy"):
                    self.clips.append(os.path.join(root, filename))
        if not self.clips:
            raise RuntimeError(f"No .npy files under {root_dir}.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip_path = self.clips[index]
        clip = np.load(clip_path).astype(np.float32)
        clip = normalize_clip_layout(clip, clip_path)
        clip = replicate_pad(clip, 128)
        print(clip_path, clip.shape)
        return clip


def coding_order_table7x7(patch_sz=64, mask_type="3P"):
    if mask_type not in ("5P", "4P", "3P", "2P", "P"):
        raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    cot = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
    if mask_type == "5P":
        for i in range(patch_sz):
            cot[i, :] = torch.arange(4 * i + 1, 4 * i + 1 + patch_sz)
    elif mask_type == "4P":
        for i in range(patch_sz):
            cot[i, :] = torch.arange(3 * i + 1, 3 * i + 1 + patch_sz)
    elif mask_type == "3P":
        for i in range(patch_sz):
            cot[i, :] = torch.arange(2 * i + 1, 2 * i + 1 + patch_sz)
    elif mask_type == "2P":
        for i in range(patch_sz):
            cot[i, :] = torch.arange(i + 1, i + 1 + patch_sz)
    elif mask_type == "P":
        for i in range(patch_sz):
            cot[i, :] = torch.arange(1, 1 + patch_sz)
    return cot


def img2patch(img, patch_sz):
    if len(img.shape) == 2:
        img = img.unsqueeze(2)
    h, w, _ = img.shape
    h_num = h // patch_sz - (patch_sz - (h % patch_sz)) // patch_sz + 1
    w_num = w // patch_sz - (patch_sz - (w % patch_sz)) // patch_sz + 1
    img_pad = nn.functional.pad(
        img,
        (0, 0, (patch_sz - w % patch_sz) % patch_sz, 0, (patch_sz - h % patch_sz) % patch_sz, 0),
    )
    patch_h = torch.chunk(img_pad, h_num, dim=0)
    patch_h = torch.stack(patch_h, dim=0)
    patch_w = torch.chunk(patch_h, w_num, dim=2)
    patch = torch.cat(patch_w, dim=0)
    return patch.permute(0, 3, 1, 2)


def patch2img(patch, img_sz):
    h, w = img_sz
    patch_sz = patch.shape[2]
    h_num = h // patch_sz - (patch_sz - (h % patch_sz)) // patch_sz + 1
    w_num = w // patch_sz - (patch_sz - (w % patch_sz)) // patch_sz + 1
    patch = patch.permute(0, 2, 3, 1)
    patch_w = torch.chunk(patch, w_num, dim=0)
    patch_h = torch.cat(patch_w, dim=2)
    patch_h = torch.chunk(patch_h, h_num, dim=0)
    img = torch.cat(patch_h, dim=1).squeeze(0)
    return img[-h:, -w:, :]


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
        new_key = new_key.replace(".entropy_bottleneck._matrix", ".entropy_bottleneck.matrices.")
        new_key = new_key.replace(".entropy_bottleneck._bias", ".entropy_bottleneck.biases.")
        new_key = new_key.replace(".entropy_bottleneck._factor", ".entropy_bottleneck.factors.")
        new_state_dict[new_key] = value
    return new_state_dict


def normalize_checkpoint_keys(checkpoint):
    """Make a state_dict loadable no matter how it was saved.

    Handles {"model_state_dict": ...} wrappers, the "module." prefix written by
    nn.DataParallel / DDP runs and the "_orig_mod." prefix of torch.compile
    runs, plus the old compressai entropy-bottleneck parameter names.
    """
    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix(state_dict, prefix="module.")
    state_dict = remove_module_prefix(state_dict, prefix="_orig_mod.")
    return convert_entropy_bottleneck_keys(state_dict)


def next_available_path(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{stem}({idx}){ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def lossy_size_bytes(code_lossy):
    """Total bytes of the lossy bitstream: every y stream plus every z stream.

    Note: for a batch of patches the hyperprior emits one z string per patch,
    so both lists must be summed in full (the previous version only counted
    the first z string and therefore under-reported the I-frame rate).
    """
    return sum(len(string) for stream in code_lossy["img_strings"] for string in stream)


def conditional_lmm_params(model, rp, ctx, tau):
    tau_cond = min(tau, model.cond_num)
    tau_list = torch.full(
        (rp.shape[0],),
        tau_cond - 1,
        dtype=torch.long,
        device=rp.device,
    )
    cond = F.one_hot(tau_list, num_classes=model.cond_num).float()
    rp_ctx = model.fusion_cond(torch.cat((rp, ctx), 1), cond)
    return model.residual_compressor_cond(rp_ctx, cond)


def compress(model, input_img, cot, tau=0, patch_sz=64, ref_l=None, ref_r=None):
    global enc_lossy_time, enc_res_time
    norm_scale = 1 / nii_max_f * 2
    half = (0.5 + tau) * norm_scale
    code_res = []
    img_shape = input_img.shape[:2]
    bin_sz = 2 * tau + 1
    samples_end = (nii_max // bin_sz) * bin_sz

    device = next(model.parameters()).device
    if ref_l is None:
        x = img2patch(torch.as_tensor(input_img), patch_sz=patch_sz).to(device)
    else:
        x = torch.as_tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)
        ref_l = torch.as_tensor(ref_l).unsqueeze(0).unsqueeze(0).to(device)
        ref_r = torch.as_tensor(ref_r).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        time_start = time.time()
        if ref_l is None:
            code_lossy = model.lossy_compressor.compress(x)
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])
        else:
            code_lossy = model.lossy_compressor.compress(x, ref_l, ref_r)
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"], ref_l, ref_r)
        time_end_lossy = time.time()

        if ref_l is not None:
            rec_lossy["x_hat"] = patchify_to_batch(rec_lossy["x_hat"], 64)
            rec_lossy["res_prior"] = patchify_to_batch(rec_lossy["res_prior"], 64)
            x = patchify_to_batch(x, 64)

        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]
        res = x - x_hat
        res_q = torch.where(
            res >= 0,
            (2 * tau + 1) * torch.floor((res + tau) / (2 * tau + 1)),
            (2 * tau + 1) * torch.ceil((res - tau) / (2 * tau + 1)),
        )

        res_q_max = int(torch.max(res_q).item())
        res_q_min = int(torch.min(res_q).item())
        res_q_max_norm = res_q_max * norm_scale
        res_q_min_norm = res_q_min * norm_scale
        samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
        samples = samples * norm_scale

        res_tmp = torch.zeros_like(x_hat)
        ctx_total = model.mask_conv(res_q * norm_scale)
        max_step = torch.max(cot)
        for i in range(max_step):
            h_idx, w_idx = torch.nonzero(cot == i + 1, as_tuple=True)
            ctx = ctx_total[:, :, h_idx, w_idx].unsqueeze(3)
            rp = res_prior[:, :, h_idx, w_idx].unsqueeze(3)
            res_crop = res_q[:, :, h_idx, w_idx]
            res_tmp_crop = res_tmp[:, :, h_idx, w_idx].unsqueeze(3)

            if tau == 0:
                rp_ctx = model.fusion(torch.cat((rp, ctx), 1))
                lmm_params = model.residual_compressor(rp_ctx)
            else:
                lmm_params = conditional_lmm_params(model, rp, ctx, tau)
            mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
            coeffs = torch.tanh(coeffs)

            mu_c = mu.permute(0, 2, 1, 3)
            samples_centered = samples - mu_c
            inv_sigma = torch.exp(-log_sigma).permute(0, 2, 1, 3)
            plus_in = inv_sigma * (samples_centered + half)
            cdf_plus = torch.sigmoid(plus_in)
            min_in = inv_sigma * (samples_centered - half)
            cdf_min = torch.sigmoid(min_in)
            cdf_delta = cdf_plus - cdf_min
            one_minus_cdf_min = torch.exp(-F.softplus(min_in))
            cdf_plus = torch.exp(plus_in - F.softplus(plus_in))
            samples2 = samples - torch.zeros_like(mu_c)
            cdf_delta = torch.where(
                samples2 - half < res_q_min_norm,
                cdf_plus,
                torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min, cdf_delta),
            )

            weights_c = weights.permute(0, 2, 1, 3)
            m = torch.amax(weights_c, 2, keepdim=True)
            weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
            pmf = torch.sum(cdf_delta * weights_c, dim=2)
            pmf = pmf.clamp_(1.0 / 54800, 1.0)
            pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
            cdf = torch.cumsum(pmf, dim=2).clamp_(0.0, 1.0)
            cdf = F.pad(cdf, (1, 0))

            symbol = torch.div(res_crop[:, 0, :].short() - res_q_min, bin_sz, rounding_mode="floor")
            res_stream = torchac.encode_float_cdf(cdf.cpu(), symbol.cpu(), needs_normalization=False, check_input_bounds=False)
            code_res.append(res_stream)
            res_tmp_crop[:, 0, :, 0] = symbol.float() * bin_sz + res_q_min
            res_tmp[:, :, h_idx, w_idx] = res_tmp_crop.squeeze(3)

        time_end_res = time.time()

    enc_lossy_time += time_end_lossy - time_start
    enc_res_time += time_end_res - time_end_lossy
    return code_lossy, code_res, img_shape, (res_q_min, res_q_max)


def decompress(model, code_lossy, code_res, img_shape, res_range, cot, tau=0, patch_sz=64, ref_l=None, ref_r=None):
    global dec_lossy_time, dec_res_time
    norm_scale = 1 / nii_max_f * 2
    half = (0.5 + tau) * norm_scale
    bin_sz = 2 * tau + 1
    device = next(model.parameters()).device

    res_q_min, res_q_max = res_range
    res_q_max_norm = res_q_max * norm_scale
    res_q_min_norm = res_q_min * norm_scale
    samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
    samples = samples * norm_scale

    if ref_l is not None:
        ref_l = torch.as_tensor(ref_l).unsqueeze(0).unsqueeze(0).to(device)
        ref_r = torch.as_tensor(ref_r).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        time_start = time.time()
        if ref_l is None:
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])
        else:
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"], ref_l, ref_r)
        time_end_lossy = time.time()

        if ref_l is not None:
            rec_lossy["x_hat"] = patchify_to_batch(rec_lossy["x_hat"], 64)
            rec_lossy["res_prior"] = patchify_to_batch(rec_lossy["res_prior"], 64)

        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]
        res_tmp = torch.zeros_like(x_hat)
        max_step = torch.max(cot)
        j = 0
        for i in range(max_step):
            h_idx, w_idx = torch.nonzero(cot == i + 1, as_tuple=True)
            ctx = model.mask_conv(res_tmp * norm_scale)[:, :, h_idx, w_idx].unsqueeze(3)
            rp = res_prior[:, :, h_idx, w_idx].unsqueeze(3)
            res_crop = res_tmp[:, :, h_idx, w_idx].unsqueeze(3)

            if tau == 0:
                rp_ctx = model.fusion(torch.cat((rp, ctx), 1))
                lmm_params = model.residual_compressor(rp_ctx)
            else:
                lmm_params = conditional_lmm_params(model, rp, ctx, tau)
            mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
            coeffs = torch.tanh(coeffs)

            mu_c = mu.permute(0, 2, 1, 3)
            samples_centered = samples - mu_c
            inv_sigma = torch.exp(-log_sigma).permute(0, 2, 1, 3)
            plus_in = inv_sigma * (samples_centered + half)
            cdf_plus = torch.sigmoid(plus_in)
            min_in = inv_sigma * (samples_centered - half)
            cdf_min = torch.sigmoid(min_in)
            cdf_delta = cdf_plus - cdf_min
            one_minus_cdf_min = torch.exp(-F.softplus(min_in))
            cdf_plus = torch.exp(plus_in - F.softplus(plus_in))
            samples2 = samples - torch.zeros_like(mu_c)
            cdf_delta = torch.where(
                samples2 - half < res_q_min_norm,
                cdf_plus,
                torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min, cdf_delta),
            )

            weights_c = weights.permute(0, 2, 1, 3)
            m = torch.amax(weights_c, 2, keepdim=True)
            weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
            pmf = torch.sum(cdf_delta * weights_c, dim=2)
            pmf = pmf.clamp_(1.0 / 54800, 1.0)
            pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
            cdf = torch.cumsum(pmf, dim=2).clamp_(0.0, 1.0)
            cdf = F.pad(cdf, (1, 0))

            symbol_out = torchac.decode_float_cdf(cdf.cpu(), code_res[j], needs_normalization=False)
            res_crop[:, 0, :, 0] = symbol_out.float() * bin_sz + res_q_min
            j += 1
            res_tmp[:, :, h_idx, w_idx] = res_crop.squeeze(3)

        time_end_res = time.time()

    dec_lossy_time += time_end_lossy - time_start
    dec_res_time += time_end_res - time_end_lossy
    x_rec = x_hat + res_tmp
    if ref_l is not None:
        x_rec = unpatchify_from_batch(x_rec, 1, img_shape[0], img_shape[1])
    else:
        x_rec = patch2img(x_rec, img_shape)
    return x_rec.clamp_(min=0, max=nii_max).squeeze()


def gop_starts(num_frames, gop_size):
    starts = list(range(0, num_frames - gop_size + 1, gop_size))
    if num_frames >= gop_size and (not starts or starts[-1] != num_frames - gop_size):
        starts.append(num_frames - gop_size)
    return sorted(set(starts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, default=0)
    parser.add_argument("--data-path", type=str, default="../Mosmed/test-full")
    parser.add_argument("--i-frame-ckp", type=str, default="ckp_nll/0001/ckp.tar")
    parser.add_argument("--p-frame-ckp", type=str, required=True)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--log-prefix", type=str, default="log-static-b-diff")
    parser.add_argument("--gop-size", type=int, default=9)
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="DDIM sampling steps of the diffusion lossy compressor (use 5 for a quick run).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Embedding width of the diffusion lossy compressor; must match the trained "
             "checkpoint (train_static_b_diff.py --diff-embed-dim).",
    )
    parser.add_argument(
        "--latent-ch",
        type=int,
        default=192,
        help="Channel count of the P-frame base-layer latent y; must match the trained "
             "checkpoint (train_static_b_diff.py --latent-ch). Default 192 is what every "
             "checkpoint so far was trained with.",
    )
    parser.add_argument("--gpu-id", "--cuda-visible-devices", dest="gpu_id", type=str, default=CUDA_VISIBLE_DEVICES)
    args = parser.parse_args()

    if args.gop_size != 9:
        raise ValueError("Static B-frame test currently supports only --gop-size 9.")

    device = torch.device("cuda")
    dataset = Dataset(args.data_path)
    model_i = NearLosslessCompressor_I(192, 5, "3P").eval().to(device)
    model_p = NearLosslessCompressor_P(
        args.latent_ch, 5, "3P", sample_steps=args.steps, embed_dim=args.embed_dim
    ).eval().to(device)

    ckp_i = torch.load(args.i_frame_ckp, map_location=device, weights_only=False)
    model_i.load_state_dict(normalize_checkpoint_keys(ckp_i))
    model_i.lossy_compressor.update(force=True)

    ckp_p = torch.load(args.p_frame_ckp, map_location=device, weights_only=False)
    model_p.load_state_dict(normalize_checkpoint_keys(ckp_p))
    model_p.lossy_compressor.update(force=True)

    cot = coding_order_table7x7()
    log_file = args.log_file or f"{args.log_prefix}-{args.tau}.txt"
    log_file = next_available_path(log_file)
    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"[info] log_file={log_file}")
    print(f"[info] diffusion ddim steps={args.steps}, embed_dim={args.embed_dim}")

    i_count = p_count = 0
    i_pn = p_pn = 0
    i_zp = p_zp = 0
    i_res_bit = p_res_bit = 0
    i_lossy_bit = p_lossy_bit = 0
    max_diff_all = 0.0

    with open(log_file, "a") as f:
        f.write(f"Lossy compressor: diff_sit_static_b (ddim_steps={args.steps})\n")
        f.write(f"Static ref_map: {STATIC_REF_MAP_9}\n")

    for clip in tqdm(dataset, desc="clips"):
        starts = gop_starts(len(clip), args.gop_size)
        for start in starts:
            gop = clip[start:start + args.gop_size]
            ref_buf = [None] * args.gop_size
            for step, (target, ref_l_idx, ref_r_idx) in enumerate(STATIC_REF_MAP_9):
                frame = gop[target]
                if step == 0:
                    code_lossy, code_res, img_shape, res_range = compress(model_i, frame, cot, args.tau)
                    rec = decompress(model_i, code_lossy, code_res, img_shape, res_range, cot, args.tau)
                else:
                    code_lossy, code_res, img_shape, res_range = compress(
                        model_p,
                        frame,
                        cot,
                        args.tau,
                        ref_l=ref_buf[ref_l_idx],
                        ref_r=ref_buf[ref_r_idx],
                    )
                    rec = decompress(
                        model_p,
                        code_lossy,
                        code_res,
                        img_shape,
                        res_range,
                        cot,
                        args.tau,
                        ref_l=ref_buf[ref_l_idx],
                        ref_r=ref_buf[ref_r_idx],
                    )

                img_ls_sz = lossy_size_bytes(code_lossy)
                res_sz = sum(len(x) for x in code_res)
                nll_sz = img_ls_sz + res_sz + 6 * 4
                rec_np = rec.detach().cpu().numpy().squeeze()
                ref_buf[target] = rec_np
                max_diff = float(np.max(np.abs(rec_np - frame)))
                max_diff_all = max(max_diff_all, max_diff)
                pixels = int(np.prod(frame.shape))

                if step == 0:
                    i_count += 1
                    i_pn += pixels
                    i_res_bit += res_sz * 8
                    i_lossy_bit += img_ls_sz * 8
                    i_zp += 6 * 4 * 8
                    label = "I"
                else:
                    p_count += 1
                    p_pn += pixels
                    p_res_bit += res_sz * 8
                    p_lossy_bit += img_ls_sz * 8
                    p_zp += 6 * 4 * 8
                    label = "B"

                line = (
                    f"{label}: clip_start={start}, target={target}, "
                    f"bpsp={nll_sz * 8 / pixels:.4f}, "
                    f"img_bpsp={img_ls_sz * 8 / pixels:.4f}, "
                    f"res_bpsp={res_sz * 8 / pixels:.4f}, "
                    f"max_error={max_diff:.0f}"
                )
                print(line)
                with open(log_file, "a") as f:
                    f.write(line + "\n")

    total_pn = i_pn + p_pn
    total_bits = i_zp + i_res_bit + i_lossy_bit + p_zp + p_res_bit + p_lossy_bit
    with open(log_file, "a") as f:
        f.write("\n===== Summary =====\n")
        f.write(f"P-frame model: {args.p_frame_ckp}\n")
        f.write(f"I-frame model: {args.i_frame_ckp}\n")
        f.write(f"Lossy compressor: diff_sit_static_b (ddim_steps={args.steps})\n")
        f.write(f"Tau: {args.tau}\n")
        f.write(f"Static ref_map: {STATIC_REF_MAP_9}\n")
        f.write(f"I-frames: count={i_count}, bpp={(i_zp + i_res_bit + i_lossy_bit) / max(i_pn, 1):.4f}\n")
        f.write(f"B-frames: count={p_count}, bpp={(p_zp + p_res_bit + p_lossy_bit) / max(p_pn, 1):.4f}\n")
        f.write(f"Total: bpp={total_bits / max(total_pn, 1):.4f}, max_error={max_diff_all:.0f}\n")
        f.write(f"Enc time: lossy={enc_lossy_time:.2f}s, res={enc_res_time:.2f}s\n")
        f.write(f"Dec time: lossy={dec_lossy_time:.2f}s, res={dec_res_time:.2f}s\n")

    print(f"[done] Results saved to {log_file}")


if __name__ == "__main__":
    main()


r'''
运行指令（工作目录：DLPR-Mosmed-nll/）
====================================================================

冒烟（5 步 DDIM，先确认能跑通、tau=0 时 max_error=0）
python test_static_b_diff.py \
  --p-frame-ckp ./log_static_b_diff/p-Ep3951-0.00000593-5.465-7.055.pth \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --data-path ../Mosmed/test-full \
  --tau 0 --steps 5 --embed-dim 64 --gpu-id 0 \
  --log-file log-static-b-diff-tau-0-steps5.txt

正式（20 步 DDIM，与 P 帧 diff-sit 的 test4-eval-diff-sit.py 默认一致）
python test_static_b_diff.py \
  --p-frame-ckp ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --data-path ../Mosmed/test-full \
  --tau 0 --steps 20 --embed-dim 64 --gpu-id 0 \
  --log-file log-static-b-diff-tau-0.txt

一次跑完 tau = 0/1/2
for tau in 0 1 2; do
  python test_static_b_diff.py \
    --p-frame-ckp ./log_static_b_diff_B/p-Ep<latest>-<img>-<img+res>-<img+res_cond>.pth \
    --i-frame-ckp ckp_nll/0001/ckp.tar \
    --data-path ../Mosmed/test-full \
    --tau $tau --steps 20 --embed-dim 64 --gpu-id 0 \
    --log-file log-static-b-diff-tau-$tau.txt
done

参数要点
  --p-frame-ckp  必填，必须是 train_static_b_diff.py 训出的扩散版 p-Ep*.pth；
                 与 log_static_b/ 的普通版 ckpt 权重键不同，两者不能混用
  --steps        DDIM 采样步数，只影响基础层重建质量与速度，不改变码流结构
  --embed-dim    扩散压缩器宽度，必须与训练时的 --diff-embed-dim 一致（默认 64）
  --tau          0 无损；1..5 近无损 max_error<=tau
  --gop-size     固定 9，脚本内会校验

  I 帧仍是普通自编码器 NearLosslessCompressor_I，只需 --i-frame-ckp。

输出
  日志格式与 test_static_b.py 完全一致（每帧 bpsp / img_bpsp / res_bpsp /
  max_error + 末尾 ===== Summary =====），头部多一行
  "Lossy compressor: diff_sit_static_b (ddim_steps=...)"，不保存重建图像。
  基础层塌缩的 ckpt 会表现为 img_bpsp≈0、码率几乎全由 res_bpsp 贡献。

排错
  若报 "decompress() takes 3 positional arguments but 4 were given"，说明模型里
  熵编码调用与当前 compressai 的签名不匹配，先自检：
  python -c "import inspect; from compressai.entropy_models import EntropyBottleneck as E; print(inspect.signature(E.decompress))"
  正常输出 (self, strings: List[bytes], size: Tuple[int, int]) -> Tensor，
  对应 nll_model_b_diff.py 里 self.entropy_bottleneck.decompress(strings, size)。
'''

