"""Initialize a NearLosslessCompressor_P (DiffusionLMMHead version) from a
baseline checkpoint that was trained with the old ResidualCompressor head.

What this does:
  1. Build the *new* NearLosslessCompressor_P (residual_compressor is now
     DiffusionLMMHead, which contains a frozen `teacher = ResidualCompressor`
     plus a randomly-init `unet = LMM_UNet`).
  2. Load the baseline ckp (old layout) with strict=False so we keep:
        lossy_compressor.*           (CDC encoder/decoder/U-Net etc.)
        mask_conv.*
        fusion.*
        residual_compressor_cond.*
        fusion_cond.*
  3. Take the old `residual_compressor.*` weights (which match the
     ResidualCompressor signature) and copy them into
     `residual_compressor.teacher.*` via DiffusionLMMHead.load_teacher().
  4. Leave residual_compressor.unet.* as freshly initialized -- Stage A
     self-distillation will train them.
  5. Save the merged state_dict so train4_diff_sit.py --resume can pick it up.

Usage:
  python init_diff_lmm_from_baseline.py \\
      --old-ckp ./log4/p-Ep541-0.179-4.892-5.286.pth \\
      --new-ckp ./log4/p_init_with_teacher.pth
"""

import argparse
import os
import torch
import torch.nn as nn

from compressai.models.utils import update_registered_buffers

from nll_model_p import NearLosslessCompressor_P


def main(old_ckp_path: str, new_ckp_path: str, num_ch: int = 192, cond_num: int = 5):
    if not os.path.isfile(old_ckp_path):
        raise FileNotFoundError(f"Baseline ckp not found: {old_ckp_path}")

    print(f"[1/4] Building NearLosslessCompressor_P (DiffusionLMMHead version) ...")
    model = NearLosslessCompressor_P(num_ch, cond_num, "3P")

    print(f"[2/4] Loading baseline ckp: {old_ckp_path}")
    old_sd = torch.load(old_ckp_path, map_location="cpu", weights_only=False)
    if isinstance(old_sd, dict) and "state_dict" in old_sd and not any(
        k.startswith(("lossy_compressor.", "mask_conv.", "residual_compressor"))
        for k in old_sd.keys()
    ):
        old_sd = old_sd["state_dict"]
    if isinstance(old_sd, dict) and "model_state_dict" in old_sd and not any(
        k.startswith(("lossy_compressor.", "mask_conv.", "residual_compressor"))
        for k in old_sd.keys()
    ):
        old_sd = old_sd["model_state_dict"]

    legacy_rc_keys = [
        k for k in old_sd.keys()
        if k.startswith("residual_compressor.")
        and not k.startswith("residual_compressor_cond.")
    ]
    if not legacy_rc_keys:
        raise RuntimeError(
            "Old ckp has no `residual_compressor.*` keys. Is this a baseline "
            "trained with the original ResidualCompressor head?"
        )
    sample_legacy = sorted(legacy_rc_keys)[:5]
    print(f"      Found {len(legacy_rc_keys)} legacy residual_compressor keys, e.g. {sample_legacy}")

    teacher_sd = {
        k[len("residual_compressor."):]: v
        for k, v in old_sd.items()
        if k.startswith("residual_compressor.")
        and not k.startswith("residual_compressor_cond.")
    }

    print(f"[3/4] Loading non-residual weights into the new model (strict=False) ...")
    remain_sd = {
        k: v for k, v in old_sd.items()
        if not (
            k.startswith("residual_compressor.")
            and not k.startswith("residual_compressor_cond.")
        )
    }
    # Bypass NearLosslessCompressor_P.load_state_dict override (which may not
    # forward strict= on older revisions of nll_model_p.py). We manually
    # replicate what the override does (resize entropy-bottleneck buffers)
    # and then call nn.Module.load_state_dict directly with strict=False.
    update_registered_buffers(
        model.lossy_compressor.gaussian_conditional,
        "lossy_compressor.gaussian_conditional",
        ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        remain_sd,
    )
    missing_keys, unexpected = nn.Module.load_state_dict(model, remain_sd, strict=False)
    missing = list(missing_keys)
    unexpected = list(unexpected)

    unexpected_non_rc = [k for k in unexpected if not k.startswith("residual_compressor.")]
    if unexpected_non_rc:
        print(f"      WARNING: unexpected non-residual keys (will be ignored): {unexpected_non_rc[:10]}")

    missing_outside_rc = [k for k in missing if not k.startswith("residual_compressor.")]
    if missing_outside_rc:
        print(f"      WARNING: missing non-residual keys (will stay at init): {missing_outside_rc[:10]}")

    n_unet = sum(1 for k in missing if k.startswith("residual_compressor.unet."))
    n_teacher = sum(1 for k in missing if k.startswith("residual_compressor.teacher."))
    n_buffers = len(missing) - n_unet - n_teacher - len(missing_outside_rc)
    print(f"      Skipped (will be filled or stay at init):"
          f" unet={n_unet}, teacher={n_teacher}, buffers/other={n_buffers}")

    print(f"[4/4] Loading baseline residual_compressor weights into DiffusionLMMHead.teacher ...")
    model.residual_compressor.load_teacher(teacher_sd)
    teacher_grad = any(p.requires_grad for p in model.residual_compressor.teacher.parameters())
    assert not teacher_grad, "teacher must be frozen after load_teacher()"

    os.makedirs(os.path.dirname(new_ckp_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), new_ckp_path)
    print(f"\nDone. Saved Stage-A starting checkpoint to:\n  {new_ckp_path}")
    print(f"\nNext: run Stage A self-distillation with this ckp as --resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ckp", type=str,
                        default="./log4/p-Ep541-0.179-4.892-5.286.pth",
                        help="Baseline ckp trained with the old ResidualCompressor head.")
    parser.add_argument("--new-ckp", type=str,
                        default="./log4/p_init_with_teacher.pth",
                        help="Output ckp for Stage A warmup.")
    parser.add_argument("--num-ch", type=int, default=192)
    parser.add_argument("--cond-num", type=int, default=5)
    args = parser.parse_args()

    main(args.old_ckp, args.new_ckp, num_ch=args.num_ch, cond_num=args.cond_num)
