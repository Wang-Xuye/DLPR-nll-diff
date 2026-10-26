import os
import argparse


_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--gpu-id", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
_bootstrap_args, _ = _bootstrap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _bootstrap_args.gpu_id

import numpy as np
import torch
from tqdm import tqdm

from global_nii_max import nii_max_f
from nll_model_eval_p import NearLosslessCompressor_I
from test import Dataset, coding_order_table7x7, compress, decompress, load_checkpoint_state


def build_output_path(prefix, tau):
    return f"{prefix}_tau{tau}.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau",
        type=int,
        default=1,
        choices=range(0, 6),
        help="near-lossless constraint, 0 to 5 (default: 1)",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=_bootstrap_args.gpu_id,
        help="Visible GPU id, for example 0 or 2",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../Mosmed/test-slice",
        help="Directory containing test sequences",
    )
    parser.add_argument(
        "--i-frame-ckp",
        type=str,
        default="/root/autodl-tmp/DLPR-Mosmed-nll/ckp_nll/0001/ckp.tar",
        help="I-frame checkpoint",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="test_all_i",
        help="Output filename prefix; tau is appended automatically",
    )
    args = parser.parse_args()

    output_path = build_output_path(args.output_prefix, args.tau)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    def log(message):
        print(message)
        output_file.write(message + "\n")
        output_file.flush()

    device = torch.device("cuda")
    nll_module_I = NearLosslessCompressor_I(192, 5).eval().to(device)

    dset = Dataset(args.data_path)
    checkpoint = torch.load(args.i_frame_ckp, map_location=device)
    load_checkpoint_state(nll_module_I, checkpoint)
    nll_module_I.lossy_compressor.update(force=True)

    COT = coding_order_table7x7()
    frame_count = 0
    total_bpp = 0.0
    total_img_bpp = 0.0
    total_res_bpp = 0.0
    max_diff = 0.0

    with open(output_path, "w", encoding="utf-8") as output_file:
        log("All frames encoded with I-frame model")
        log(f"tau: {args.tau}")
        log(f"data_path: {args.data_path}")
        log(f"i_frame_ckp: {args.i_frame_ckp}")
        log(f"output_file: {output_path}")

        for filez in tqdm(dset):
            clip_bpp = 0.0
            clip_frame_count = 0

            for frame_idx, frame in enumerate(filez):
                code_lossy, code_res, img_shape, res_range = compress(
                    nll_module_I,
                    frame,
                    COT,
                    args.tau,
                    64,
                    5,
                    None,
                )

                img_ls_sz = sum(
                    len(code_lossy["img_strings"][0][j])
                    + len(code_lossy["img_strings"][1][j])
                    for j in range(len(code_lossy["img_strings"][0]))
                )
                res_sz = sum(len(code) for code in code_res)
                nll_sz = img_ls_sz + res_sz + 6 * 4

                reconstructed = decompress(
                    nll_module_I,
                    code_lossy,
                    code_res,
                    img_shape,
                    res_range,
                    COT,
                    args.tau,
                    5,
                    64,
                    None,
                )
                reconstructed = reconstructed.cpu().numpy().squeeze()

                frame_pixels = np.prod(frame.shape)
                frame_max_diff = float(np.max(np.abs(reconstructed - frame)))
                mse_loss = float(np.mean((frame - reconstructed) ** 2))
                psnr = float("inf") if mse_loss == 0 else 10.0 * np.log10(nii_max_f ** 2 / mse_loss)

                bpp = nll_sz * 8 / frame_pixels
                img_bpp = img_ls_sz * 8 / frame_pixels
                res_bpp = res_sz * 8 / frame_pixels

                frame_count += 1
                clip_frame_count += 1
                clip_bpp += bpp
                total_bpp += bpp
                total_img_bpp += img_bpp
                total_res_bpp += res_bpp
                max_diff = max(max_diff, frame_max_diff)

                log(
                    "frame: {}, shape: {}, bpsp: {:.4f}, img_bpsp: {:.4f}, "
                    "res_bpsp: {:.4f}, size: {:.0f}, max_diff: {:.4f}, psnr: {:.4f}".format(
                        frame_idx,
                        tuple(frame.shape),
                        bpp,
                        img_bpp,
                        res_bpp,
                        nll_sz * 8,
                        frame_max_diff,
                        psnr,
                    )
                )

            log(
                "clip average: bpsp: {:.4f}, frames: {}".format(
                    clip_bpp / clip_frame_count if clip_frame_count else 0.0,
                    clip_frame_count,
                )
            )

        log("summary:")
        log(f"frames: {frame_count}")
        log(f"average bpsp: {total_bpp / frame_count if frame_count else 0.0:.4f}")
        log(f"average img_bpsp: {total_img_bpp / frame_count if frame_count else 0.0:.4f}")
        log(f"average res_bpsp: {total_res_bpp / frame_count if frame_count else 0.0:.4f}")
        log(f"max_diff: {max_diff:.4f}")


if __name__ == "__main__":
    main()

'''
python test_all_i.py \
  --tau 1 \
  --gpu-id 0 \
  --data-path ../Chaos_nii/test-slice \
  --i-frame-ckp ./ckp_nll/dlpr_chaos_baseline/p-Ep750-0.353-5.810-6.004.pth \
  --output-prefix ./results/chaos_all_i

python test_all_i.py \
  --tau 2 \
  --gpu-id 1 \
  --data-path ../Chaos_nii/test-slice \
  --i-frame-ckp ./ckp_nll/dlpr_chaos_baseline/p-Ep750-0.353-5.810-6.004.pth \
  --output-prefix ./results/chaos_all_i

python test_all_i.py \
  --tau 3 \
  --gpu-id 2 \
  --data-path ../Chaos_nii/test-slice \
  --i-frame-ckp ./ckp_nll/dlpr_chaos_baseline/p-Ep750-0.353-5.810-6.004.pth \
  --output-prefix ./results/chaos_all_i

python test_all_i.py \
  --tau 4 \
  --gpu-id 3 \
  --data-path ../Chaos_nii/test-slice \
  --i-frame-ckp ./ckp_nll/dlpr_chaos_baseline/p-Ep750-0.353-5.810-6.004.pth \
  --output-prefix ./results/chaos_all_i
'''