"""
Bit-exact reproducibility verification for LossyCompressor_P_diff_sit.

Three modes:
  --mode same      run forward N times in the same process and compare each
                   pair (default; this proves intra-process determinism)
  --mode save      run forward once and save x_hat / res_prior / likelihoods
                   to a reference .pt file
  --mode check     run forward once and compare against a saved reference
                   .pt file (run after --mode save in a NEW process to test
                   cross-process determinism, which is the realistic
                   encoder/decoder scenario)

Usage:
  cd D:/研究生/医学影像压缩/linc/DLPR-Mosmed-nll

  # 1) intra-process (existing behavior)
  python verify_bitexact.py --ckp <ckp.pth>

  # 2) cross-process: run twice in two separate Python invocations
  python verify_bitexact.py --ckp <ckp.pth> --mode save  --ref-file ref.pt
  python verify_bitexact.py --ckp <ckp.pth> --mode check --ref-file ref.pt

Optional:
  --runs 3          # how many forward passes (mode=same only)
  --device cuda     # cuda or cpu
  --strict          # enable torch.use_deterministic_algorithms(True)
  --height/--width  # input spatial size (default 512)
  --seed N          # input RNG seed (must match across save/check runs)
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from nll_model_p import NearLosslessCompressor_P
from global_nii_max import nii_max_f


def set_deterministic(seed: int = 42, strict: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[warn] use_deterministic_algorithms failed: {e}")


def load_p_frame_model(ckp_path: str, device: str):
    model = NearLosslessCompressor_P(192, 5, "3P").to(device)
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
    print(f"[info] loading checkpoint: {ckp_path}")
    ckp = torch.load(ckp_path, map_location=device, weights_only=False)
    state = ckp.get("model_state_dict", ckp) if isinstance(ckp, dict) else ckp

    try:
        missing, unexpected = torch.nn.Module.load_state_dict(model, state, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)} (showing first 5) -> {missing[:5]}")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")
    except Exception as e:
        print(f"[warn] non-strict load failed ({e}); falling back to model.load_state_dict(state)")
        model.load_state_dict(state)
    model.eval()
    return model


def make_inputs(device: str, h: int = 512, w: int = 512, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_cpu = (torch.rand(1, 1, h, w, generator=g) * nii_max_f).round()
    ref_cpu = (torch.rand(1, 1, h, w, generator=g) * nii_max_f).round()
    return x_cpu.to(device), ref_cpu.to(device)


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.detach().cpu()
    b = b.detach().cpu()
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    nz = (diff > 0).sum().item()
    total = diff.numel()
    equal = torch.equal(a, b)
    flag = "OK " if equal else "FAIL"
    print(
        f"  [{flag}] {name:20s}  shape={tuple(a.shape)}  "
        f"max|d|={max_abs:.3e}  mean|d|={mean_abs:.3e}  "
        f"nonzero={nz}/{total}  bitwise_equal={equal}"
    )
    return equal


def run_once(model, x, ref):
    inner = model.lossy_compressor
    with torch.no_grad():
        out = inner(x, ref)
    return {key: (val.detach().cpu().clone() if torch.is_tensor(val) else val)
            for key, val in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp", required=True, help="path to P-frame checkpoint")
    parser.add_argument("--mode", choices=["same", "save", "check"], default="same",
                        help="same=N runs in this process; save=write reference .pt; "
                             "check=compare against reference .pt")
    parser.add_argument("--ref-file", default="verify_ref.pt",
                        help="reference file path for save/check modes")
    parser.add_argument("--runs", type=int, default=3, help="forward passes (mode=same)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict", action="store_true",
                        help="use_deterministic_algorithms(True)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0,
                        help="input RNG seed (MUST match across save and check)")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print(f"Bit-exact verification (mode={args.mode})")
    print("=" * 70)
    print(f"  device     : {args.device}")
    print(f"  ckp        : {args.ckp}")
    print(f"  ref-file   : {args.ref_file}")
    print(f"  seed       : {args.seed}")
    print(f"  spatial    : {args.height}x{args.width}")
    print(f"  strict det : {args.strict}")
    print()

    set_deterministic(seed=args.seed, strict=args.strict)
    model = load_p_frame_model(args.ckp, args.device)
    inner = model.lossy_compressor
    print(f"  inner type : {type(inner).__name__}")
    print(f"  training?  : {model.training} / {inner.training}  (must be False)")
    print()

    x, ref = make_inputs(args.device, args.height, args.width, seed=args.seed)
    print(f"  input shape: {tuple(x.shape)}  ref shape: {tuple(ref.shape)}")
    print(f"  input dtype: {x.dtype}  range=[{x.min():.1f}, {x.max():.1f}]")
    print()

    keys_to_check = ["x_hat", "res_prior"]

    if args.mode == "same":
        outs = []
        for k in range(args.runs):
            print(f"[run {k + 1}/{args.runs}] forward...")
            outs.append(run_once(model, x, ref))
        print()
        print("-" * 70)
        print("Pairwise comparison vs run 1")
        print("-" * 70)
        all_ok = True
        for k in range(1, args.runs):
            print(f"\nrun 1  vs  run {k + 1}:")
            for key in keys_to_check:
                ok = compare_tensors(key, outs[0][key], outs[k][key])
                all_ok = all_ok and ok
            for sub in ("y", "z"):
                ok = compare_tensors(f"likelihoods.{sub}",
                                     outs[0]["likelihoods"][sub],
                                     outs[k]["likelihoods"][sub])
                all_ok = all_ok and ok
        _print_verdict(all_ok, args.mode)

    elif args.mode == "save":
        print("[save] running forward and saving reference...")
        out = run_once(model, x, ref)
        payload = {
            "x_hat": out["x_hat"],
            "res_prior": out["res_prior"],
            "likelihoods_y": out["likelihoods"]["y"],
            "likelihoods_z": out["likelihoods"]["z"],
            "meta": {"seed": args.seed, "h": args.height, "w": args.width,
                     "device": args.device, "ckp": os.path.abspath(args.ckp)},
        }
        torch.save(payload, args.ref_file)
        print(f"[save] wrote {args.ref_file}")
        print(f"       x_hat sum    = {out['x_hat'].double().sum().item():.6f}")
        print(f"       res_prior sum= {out['res_prior'].double().sum().item():.6f}")
        print()
        print("Now run the SAME command with --mode check in a NEW Python process.")
        sys.exit(0)

    else:  # check
        if not os.path.isfile(args.ref_file):
            print(f"[error] reference file not found: {args.ref_file}")
            print("        run --mode save first")
            sys.exit(2)
        print(f"[check] loading reference: {args.ref_file}")
        ref_payload = torch.load(args.ref_file, map_location="cpu", weights_only=False)
        meta = ref_payload.get("meta", {})
        if meta.get("seed") != args.seed:
            print(f"[warn] seed mismatch: ref={meta.get('seed')} vs current={args.seed}")
        if (meta.get("h"), meta.get("w")) != (args.height, args.width):
            print(f"[warn] size mismatch: ref={meta.get('h')}x{meta.get('w')} "
                  f"vs current={args.height}x{args.width}")
        print()

        print("[check] running forward in current process...")
        cur = run_once(model, x, ref)
        print()
        print("-" * 70)
        print("Reference  vs  current process forward")
        print("-" * 70)
        all_ok = True
        all_ok &= compare_tensors("x_hat", ref_payload["x_hat"], cur["x_hat"])
        all_ok &= compare_tensors("res_prior", ref_payload["res_prior"], cur["res_prior"])
        all_ok &= compare_tensors("likelihoods.y",
                                  ref_payload["likelihoods_y"], cur["likelihoods"]["y"])
        all_ok &= compare_tensors("likelihoods.z",
                                  ref_payload["likelihoods_z"], cur["likelihoods"]["z"])
        _print_verdict(all_ok, args.mode)


def _print_verdict(all_ok: bool, mode: str) -> None:
    print()
    print("=" * 70)
    if all_ok:
        if mode == "same":
            print("RESULT: PASS  (intra-process bit-exact)")
        else:
            print("RESULT: PASS  (cross-process bit-exact)")
            print("        -> encoder/decoder split is safe; c_i is reproducible")
    else:
        print("RESULT: FAIL  (outputs differ)")
        print("        -> must locate the source of nondeterminism")
        print("        Common culprits to inspect:")
        print("          1. .train() vs .eval() state on a submodule")
        print("          2. cuDNN nondeterministic algos (try --strict)")
        print("          3. AMP / fp16 / TF32 enabled somewhere")
        print("          4. ddim_sample numerics across CUDA contexts")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
