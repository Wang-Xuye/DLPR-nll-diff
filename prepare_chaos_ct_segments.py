import argparse
from pathlib import Path

import numpy as np


def find_nii_files(input_dir):
    input_dir = Path(input_dir)
    files = []
    for path in input_dir.rglob("*"):
        name = path.name.lower()
        if path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz")):
            files.append(path)
    return sorted(files)


def strip_nii_suffix(path):
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def infer_slice_axis(shape, slice_axis=None):
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D CT volume, got shape={shape}.")
    if slice_axis is not None:
        if slice_axis < 0 or slice_axis >= len(shape):
            raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}.")
        return slice_axis
    return int(np.argmin(shape))


def volume_to_slices_first(volume, slice_axis=None):
    axis = infer_slice_axis(volume.shape, slice_axis)
    return np.moveaxis(volume, axis, 0)


def clip_to_uint16(volume, value_min=0, value_max=4095):
    volume = np.nan_to_num(volume, nan=value_min, posinf=value_max, neginf=value_min)
    volume = np.rint(volume)
    volume = np.clip(volume, value_min, value_max)
    return volume.astype(np.uint16)


def segment_starts(num_slices, segment_length=8):
    if num_slices < segment_length:
        raise ValueError(
            f"Need at least {segment_length} slices, got {num_slices}."
        )
    starts = list(range(0, num_slices - segment_length + 1, segment_length))
    last_start = num_slices - segment_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_segments(slices_first, segment_length=8):
    for start in segment_starts(slices_first.shape[0], segment_length):
        yield start, np.ascontiguousarray(slices_first[start:start + segment_length])


def load_nii(path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "Reading .nii files requires nibabel. Install it with: pip install nibabel"
        ) from exc
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj)


def relative_output_dir(input_file, input_dir, output_dir):
    relative_parent = input_file.parent.relative_to(input_dir)
    return output_dir / relative_parent


def convert_one_file(input_file, input_dir, output_dir, segment_length, slice_axis):
    volume = load_nii(input_file)
    slices_first = volume_to_slices_first(volume, slice_axis=slice_axis)
    slices_first = clip_to_uint16(slices_first)

    out_dir = relative_output_dir(input_file, input_dir, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = strip_nii_suffix(input_file)
    count = 0
    for count, (start, segment) in enumerate(
        iter_segments(slices_first, segment_length=segment_length), start=1
    ):
        out_name = f"{stem}_seg{count:04d}_s{start:04d}.npy"
        np.save(out_dir / out_name, segment)
    return count, slices_first.shape


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert CHAOS CT .nii volumes into uint16 .npy segments. "
            "Each saved segment has shape (segment_length, H, W)."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing original .nii or .nii.gz CT volumes.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where .npy segments will be saved.",
    )
    parser.add_argument(
        "--segment-length",
        default=8,
        type=int,
        help="Number of CT slices per segment. Default: 8.",
    )
    parser.add_argument(
        "--slice-axis",
        default=None,
        type=int,
        choices=(0, 1, 2),
        help="Axis used as the slice/time dimension. Default: infer as the smallest dimension.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.segment_length <= 0:
        raise ValueError("--segment-length must be positive.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    nii_files = find_nii_files(input_dir)
    if not nii_files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found under: {input_dir}")

    total_segments = 0
    skipped = 0
    for nii_file in nii_files:
        try:
            num_segments, shape = convert_one_file(
                nii_file,
                input_dir,
                output_dir,
                args.segment_length,
                args.slice_axis,
            )
        except ValueError as exc:
            skipped += 1
            print(f"[skip] {nii_file}: {exc}")
            continue
        total_segments += num_segments
        print(f"[ok] {nii_file} -> {num_segments} segments, slices-first shape={shape}")

    print(
        f"Done. files={len(nii_files)}, skipped={skipped}, "
        f"segments={total_segments}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
