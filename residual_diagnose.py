"""
Residual distribution diagnosis for the LMM-based entropy model.

Goal: answer the empirical question
  "Does the Logistic Mixture Model (LMM) actually fit the residual r = x - x_hat
   well enough, or does it leave structured bpp on the table?"

What this script computes (lossless setting, tau=0):
  1. Aggregate residual histogram (linear + log y) over many P frames.
  2. The same histogram with one fitted LMM PMF curve overlaid (mean params).
  3. Per-pixel NLL distribution under the LMM (in bits/pixel).
  4. NLL stratified by predicted sigma decile -- shows whether LMM
     systematically over/under-spends bits in low/high uncertainty regions.
  5. NLL stratified by actual |residual| decile -- shows whether tail
     residuals are paying disproportionate bits (suggests heavier tail
     than logistic mixture can model).

Outputs:
  - residual_diag_hist.png   linear/log histograms with LMM curve
  - residual_diag_nll.png    per-pixel NLL histogram + sigma/abs(r) deciles
  - residual_diag.csv        machine-readable summary

Usage:
  cd D:/研究生/医学影像压缩/linc/DLPR-Mosmed-nll

  python residual_diagnose.py \
      --p-ckp log4_diff_warmup_nobpp200/p-Ep1251-0.005-4.796-5.067.pth \
      --i-ckp ckp_nll/0001/ckp.tar \
      --data-dir <Mosmed_test_npy_dir> \
      --num-seqs 5

Notes:
  - Only P frames are diagnosed (I-frames go through pretrained DLPR).
  - Uses GOP=8 (I, P, P, ..., P), matching test4-eval-diff-sit.
  - sample_steps=20 by default (matches eval_p default).
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nll_model_eval_p import NearLosslessCompressor_I, NearLosslessCompressor_P
import nll_model_eval_p as _nme
from logisticmixturemodel import LogisticMixtureModel
from global_nii_max import nii_max_f
from utils.data.datasets import ImageDataset


def load_models(p_ckp, i_ckp, device):
    nll_I = NearLosslessCompressor_I(192, 5).eval().to(device)
    nll_P = NearLosslessCompressor_P(192, 5).eval().to(device)

    ckp_i = torch.load(i_ckp, map_location=device, weights_only=False)
    nll_I.load_state_dict(ckp_i["model_state_dict"])
    nll_I.lossy_compressor.update(force=True)

    ckp_p = torch.load(p_ckp, map_location=device, weights_only=False)
    state = ckp_p.get("model_state_dict", ckp_p) if isinstance(ckp_p, dict) else ckp_p
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    import re
    state = {re.sub(r"matrices\.(\d+)", r"_matrix\1",
            re.sub(r"biases\.(\d+)", r"_bias\1",
            re.sub(r"factors\.(\d+)", r"_factor\1", k))): v for k, v in state.items()}
    nll_P.load_state_dict(state)
    nll_P.lossy_compressor.update(force=True)
    return nll_I, nll_P


def to_bchw(x_np, device, mult=256):
    t = torch.from_numpy(np.ascontiguousarray(x_np)).float().to(device)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    h, w = t.shape[-2], t.shape[-1]
    pad_h = (mult - h % mult) % mult
    pad_w = (mult - w % mult) % mult
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
    return t, (h, w)


def collect_residuals(nll_P, sequences, device, max_p_frames):
    """Run P-frame model on (frame_i, frame_{i-1}) pairs, collect raw residuals
    and the LMM parameters predicted at each pixel."""
    residuals = []
    mu_all, log_sigma_all, weights_all, coeffs_all = [], [], [], []
    p_count = 0

    with torch.no_grad():
        for seq_idx, seq in enumerate(sequences):
            seq = np.asarray(seq).astype(np.float32)
            if seq.ndim == 3:
                a, b, c = seq.shape
                if c < a and c < b:
                    seq = np.transpose(seq, (2, 0, 1))
                    print(f"[seq {seq_idx + 1}] transposed (H,W,N)->(N,H,W): "
                          f"({a},{b},{c}) -> {seq.shape}")
            n_frames = seq.shape[0]
            sample_h, sample_w = seq.shape[-2], seq.shape[-1]
            if min(sample_h, sample_w) < 64:
                print(f"[seq {seq_idx + 1}/{len(sequences)}] skipped "
                      f"(too small: {sample_h}x{sample_w})")
                continue
            print(f"[seq {seq_idx + 1}/{len(sequences)}] frames={n_frames} "
                  f"size={sample_h}x{sample_w}")

            for i in range(0, n_frames, 8):
                gop_end = min(i + 8, n_frames)
                last_frame_t = None
                orig_hw = None
                for j in range(i, gop_end):
                    cur, hw = to_bchw(seq[j], device)
                    orig_hw = hw
                    if j == i:
                        last_frame_t = cur
                        continue

                    out = nll_P.lossy_compressor(cur, last_frame_t)
                    x_hat = out["x_hat"]
                    res_prior = out["res_prior"]

                    h, w = orig_hw
                    cur_c = cur[..., :h, :w]
                    x_hat_c = x_hat[..., :h, :w]
                    res = cur_c - x_hat_c
                    res_n = res / nii_max_f * 2

                    res_prior_c = res_prior[..., :h, :w]
                    context = nll_P.mask_conv(res_n)
                    rp_ctx = nll_P.fusion(torch.cat((res_prior_c, context), 1))
                    lmm_params = nll_P.residual_compressor(rp_ctx)
                    mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                    coeffs = torch.tanh(coeffs)

                    residuals.append(res.detach().cpu().reshape(-1))
                    mu_all.append(mu.detach().cpu().reshape(5, -1))
                    log_sigma_all.append(log_sigma.detach().cpu().reshape(5, -1))
                    weights_all.append(weights.detach().cpu().reshape(5, -1))
                    coeffs_all.append(coeffs.detach().cpu().reshape(5, -1))

                    last_frame_t = x_hat.clamp_(0, nii_max_f).detach()
                    p_count += 1
                    if p_count >= max_p_frames:
                        return _stack(residuals, mu_all, log_sigma_all,
                                      weights_all, coeffs_all)

    return _stack(residuals, mu_all, log_sigma_all, weights_all, coeffs_all)


def _stack(res, mu, log_sigma, weights, coeffs):
    return (
        torch.cat(res, dim=0),
        torch.cat(mu, dim=1),
        torch.cat(log_sigma, dim=1),
        torch.cat(weights, dim=1),
        torch.cat(coeffs, dim=1),
    )


def lmm_nll_per_pixel(res, mu, log_sigma, weights, coeffs):
    """Compute per-pixel NLL (bits) under LMM for a flattened set of pixels."""
    K, N = mu.shape
    res_n = res / nii_max_f * 2
    res_n = res_n.view(1, 1, 1, N)
    mu = mu.view(1, K, 1, N)
    log_sigma = log_sigma.view(1, K, 1, N)
    weights = weights.view(1, K, 1, N)
    coeffs = coeffs.view(1, K, 1, N)
    lmm = LogisticMixtureModel(mu, log_sigma, weights, coeffs)
    log_lik = lmm(res_n)
    nll_bits = -log_lik / math.log(2)
    return nll_bits.view(-1)


def fit_logistic_pmf_avg(mu, log_sigma, weights, coeffs, value_range, sample_n=200_000):
    """Use a random subset of pixels and average their LMM PMFs to get a
    representative 'mean' LMM curve to overlay on the histogram."""
    K, N = mu.shape
    if N > sample_n:
        idx = torch.randperm(N)[:sample_n]
        mu = mu[:, idx]; log_sigma = log_sigma[:, idx]
        weights = weights[:, idx]; coeffs = coeffs[:, idx]
        N = sample_n
    sigma = torch.exp(log_sigma)
    w = torch.softmax(weights, dim=0)

    vals = torch.arange(value_range[0], value_range[1] + 1, dtype=torch.float32)
    vals_n = vals / nii_max_f * 2
    pmf = torch.zeros_like(vals)
    for v_i, v in enumerate(vals_n):
        x = v.expand(K, N)
        plus = (x - mu + 1.0 / nii_max_f) / sigma
        minus = (x - mu - 1.0 / nii_max_f) / sigma
        cdf_plus = torch.sigmoid(plus)
        cdf_minus = torch.sigmoid(minus)
        per_k = (cdf_plus - cdf_minus).clamp_(min=1e-12)
        pmf[v_i] = (w * per_k).sum(0).mean().item()
    pmf = pmf / pmf.sum()
    return vals.numpy(), pmf.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-ckp", required=True)
    ap.add_argument("--i-ckp", required=True)
    ap.add_argument("--data-dir", required=True, help="dir with .npy sequences")
    ap.add_argument("--num-seqs", type=int, default=5)
    ap.add_argument("--max-p-frames", type=int, default=40,
                    help="cap total P frames to keep memory bounded")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-prefix", default="residual_diag")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    _nme.EVAL_DDIM_STEPS = args.steps
    print(f"[info] EVAL_DDIM_STEPS = {args.steps}")
    print(f"[info] device          = {args.device}")

    nll_I, nll_P = load_models(args.p_ckp, args.i_ckp, args.device)

    dset = ImageDataset(args.data_dir)
    sequences = [dset[i] for i in range(min(args.num_seqs, len(dset)))]

    print("[info] running P-frame forward to collect residuals...")
    res, mu, log_sigma, weights, coeffs = collect_residuals(
        nll_P, sequences, args.device, args.max_p_frames)
    print(f"[info] collected pixels: {res.numel():,}")

    res_np = res.numpy()
    abs_med = np.median(np.abs(res_np))
    abs_99 = np.quantile(np.abs(res_np), 0.99)
    abs_max = np.max(np.abs(res_np))
    print(f"[info] |res| median={abs_med:.2f} q99={abs_99:.2f} max={abs_max:.2f}")

    range_lo = int(-abs_99 * 1.2)
    range_hi = int(abs_99 * 1.2)
    print(f"[info] histogram range: [{range_lo}, {range_hi}]")

    bins = np.arange(range_lo - 0.5, range_hi + 1.5, 1.0)
    hist, edges = np.histogram(res_np, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    pmf_emp = hist / hist.sum()

    print("[info] computing average LMM PMF curve...")
    vals_lmm, pmf_lmm = fit_logistic_pmf_avg(mu, log_sigma, weights, coeffs,
                                              (range_lo, range_hi),
                                              sample_n=200_000)

    print("[info] computing per-pixel NLL...")
    sub_n = min(2_000_000, res.numel())
    idx = torch.randperm(res.numel())[:sub_n]
    nll_bits = lmm_nll_per_pixel(res[idx], mu[:, idx], log_sigma[:, idx],
                                 weights[:, idx], coeffs[:, idx])
    nll_np = nll_bits.numpy()
    print(f"[info] mean NLL = {nll_np.mean():.4f} bpp  median = {np.median(nll_np):.4f} bpp")

    sigma_eff = torch.exp(log_sigma[:, idx]).mean(0).numpy()
    sig_q = np.quantile(sigma_eff, np.linspace(0, 1, 11))
    sig_decile_nll = []
    for k in range(10):
        m = (sigma_eff >= sig_q[k]) & (sigma_eff <= sig_q[k + 1])
        sig_decile_nll.append(nll_np[m].mean() if m.sum() else np.nan)

    abs_r = np.abs(res_np[idx.numpy()])
    abs_q = np.quantile(abs_r, np.linspace(0, 1, 11))
    absr_decile_nll = []
    for k in range(10):
        m = (abs_r >= abs_q[k]) & (abs_r <= abs_q[k + 1])
        absr_decile_nll.append(nll_np[m].mean() if m.sum() else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(centers, pmf_emp, width=1.0, alpha=0.6, label="empirical")
    ax.plot(vals_lmm, pmf_lmm, "r-", lw=1.5, label="LMM (avg over pixels)")
    ax.set_xlabel("residual"); ax.set_ylabel("probability")
    ax.set_title("Residual distribution (linear y)")
    ax.legend(); ax.set_xlim(range_lo, range_hi)
    ax = axes[1]
    ax.bar(centers, pmf_emp, width=1.0, alpha=0.6, label="empirical")
    ax.plot(vals_lmm, pmf_lmm, "r-", lw=1.5, label="LMM (avg)")
    ax.set_yscale("log")
    ax.set_xlabel("residual"); ax.set_ylabel("probability (log)")
    ax.set_title("Residual distribution (log y) -- check tails")
    ax.legend(); ax.set_xlim(range_lo, range_hi)
    plt.tight_layout()
    out1 = f"{args.out_prefix}_hist.png"
    plt.savefig(out1, dpi=150); plt.close()
    print(f"[OK] saved {out1}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.hist(np.clip(nll_np, 0, 20), bins=120)
    ax.set_xlabel("per-pixel NLL (bits)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-pixel NLL distribution (mean={nll_np.mean():.3f} bpp)")
    ax = axes[1]
    ax.plot(range(1, 11), sig_decile_nll, "o-")
    ax.set_xlabel("predicted sigma decile (low -> high)")
    ax.set_ylabel("mean NLL (bits)")
    ax.set_title("NLL by predicted sigma decile")
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.plot(range(1, 11), absr_decile_nll, "o-")
    ax.set_xlabel("|residual| decile (small -> large)")
    ax.set_ylabel("mean NLL (bits)")
    ax.set_title("NLL by |residual| decile")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = f"{args.out_prefix}_nll.png"
    plt.savefig(out2, dpi=150); plt.close()
    print(f"[OK] saved {out2}")

    out3 = f"{args.out_prefix}.csv"
    with open(out3, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["pixels", res.numel()])
        w.writerow(["abs_res_median", float(abs_med)])
        w.writerow(["abs_res_q99", float(abs_99)])
        w.writerow(["abs_res_max", float(abs_max)])
        w.writerow(["mean_nll_bits", float(nll_np.mean())])
        w.writerow(["median_nll_bits", float(np.median(nll_np))])
        for k, v in enumerate(sig_decile_nll, 1):
            w.writerow([f"sigma_decile_{k}_meanNLL", float(v) if not np.isnan(v) else ""])
        for k, v in enumerate(absr_decile_nll, 1):
            w.writerow([f"absres_decile_{k}_meanNLL", float(v) if not np.isnan(v) else ""])
    print(f"[OK] saved {out3}")

    print()
    print("=" * 60)
    print("How to read the outputs:")
    print("  hist.png   linear+log histograms with LMM curve overlaid.")
    print("    Tight match -> LMM already strong. Visible mismatch in")
    print("    log plot tails -> LMM under-models heavy tails.")
    print("  nll.png    per-pixel NLL distribution and decile plots.")
    print("    A flat NLL across sigma deciles is healthy. A monotone")
    print("    rise in last 1-2 |residual| deciles flags tail bpp leakage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
