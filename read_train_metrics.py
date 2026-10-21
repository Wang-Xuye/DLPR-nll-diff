"""Read the eval metrics of a training run out of its tensorboard event files.

Stage A of train_static_b_diff.py optimises the diffusion base layer only, so the
img_bpp/res_bpp numbers in the checkpoint names are meaningless there (they are
just clamped at --likelihood-lower-bound). This script prints the eval/mse_loss
curve that actually says whether the warm-up is still improving, and maps the best
epoch back to the matching p-Ep{N}-...pth file.

Usage:
  python read_train_metrics.py --log-dir logs_static_b_diff_A --save-dir log_static_b_diff_A
  python read_train_metrics.py --log-dir logs_static_b_diff_B --last 30
  python read_train_metrics.py --log-dir logs_static_b_diff_A --tag eval/mse_loss
"""

import argparse
import glob
import os
import re


DEFAULT_TAGS = (
    "eval/mse_loss",
    "eval/img_bpp",
    "eval/res_bpp",
    "eval/res_bpp_cond",
    "eval/bpp",
    "eval/bpp_cond",
)


def load_scalars(log_dir):
    """Merge every event file in log_dir into {tag: {step: value}}.

    A run that was resumed writes a new event file, so later files override the
    same (tag, step) pair; steps are the 0-based epoch index used by eval_epoch().
    """
    files = sorted(
        glob.glob(os.path.join(log_dir, "events.out.tfevents.*")),
        key=os.path.getmtime,
    )
    if not files:
        raise SystemExit(
            f"no events.out.tfevents.* under {log_dir} "
            f"(is --log-dir the directory passed as --log-dir to the trainer?)"
        )

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as err:  # tensorboard ships with torch's tensorboard support
        raise SystemExit(
            "tensorboard is required to read the event files "
            "(pip install tensorboard)\n" + str(err)
        )

    merged = {}
    for path in files:
        accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            series = merged.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                series[event.step] = event.value
    return merged, files


def match_checkpoints(save_dir, epochs):
    """Map epoch (1-based, as in p-Ep{N}-) -> checkpoint path."""
    found = {}
    if not save_dir or not os.path.isdir(save_dir):
        return found
    for name in sorted(os.listdir(save_dir)):
        match = re.match(r"p-Ep(\d+)-", name)
        if match:
            found[int(match.group(1))] = os.path.join(save_dir, name)
    return found


def pick_stage_a(scalars, tol):
    """Apply the documented Stage A selection rule.

    1. keep the eval points whose mse is within `tol` of the best (the plateau);
    2. among them prefer the lowest img_bpp (least saturated hyperprior - Stage A
       never trains the rate side, so img_bpp drifts towards its clamp ceiling);
    3. then the lowest mse, then the earliest epoch.
    """
    mse = scalars.get("eval/mse_loss")
    if not mse:
        raise SystemExit("eval/mse_loss not found, cannot rank Stage A checkpoints")
    img = scalars.get("eval/img_bpp", {})
    best = min(mse.values())
    plateau = sorted(step for step in mse if mse[step] <= best * (1.0 + tol))

    def key(step):
        # round img_bpp the way the checkpoint name does, so sub-1% jitter does not
        # dominate the ordering
        return (round(float(img.get(step, float("inf"))), 2), mse[step], step)

    chosen = min(plateau, key=key)

    print()
    print(f"[pick] Stage A candidates inside the mse plateau (tolerance {tol * 100:.0f}%):")
    print(f"{'epoch':>6}  {'eval/mse_loss':>15}  {'eval/img_bpp':>13}  chosen")
    for step in plateau:
        img_value = img.get(step)
        img_text = f"{img_value:>13.4g}" if img_value is not None else f"{'n/a':>13}"
        mark = "  <=" if step == chosen else ""
        print(f"{step + 1:>6}  {mse[step]:>15.6g}  {img_text}{mark}")
    print(
        f"[pick] -> epoch {chosen + 1}: lowest img_bpp inside the plateau, "
        f"then lowest mse (img_bpp={float(img.get(chosen, float('nan'))):.4g}, "
        f"mse={mse[chosen]:.6g})"
    )
    return chosen


def fmt(value, width=13):
    return " " * width if value is None else f"{value:.6g}".rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True,
                        help="TensorBoard log directory of the run (logs_static_b_diff_A ...)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Checkpoint directory, used to map the best epoch to its p-Ep*.pth")
    parser.add_argument("--tag", type=str, default="eval/mse_loss",
                        help="Tag used to pick the best epoch (default eval/mse_loss)")
    parser.add_argument("--last", type=int, default=0,
                        help="Only print the last N eval points (0 = all)")
    parser.add_argument("--rank-by", type=str, default="min",
                        choices=["min", "max"],
                        help="Whether a lower or higher value of --tag is better")
    parser.add_argument("--pick-stage-a", action="store_true",
                        help="Rank Stage A checkpoints with the documented rule: mse "
                             "plateau, then lowest img_bpp, then lowest mse, then "
                             "earliest epoch")
    parser.add_argument("--plateau-tol", type=float, default=0.10,
                        help="Relative mse tolerance that defines the plateau "
                             "(default 0.10 = within 10%% of the best)")
    args = parser.parse_args()

    scalars, files = load_scalars(args.log_dir)
    print(f"[info] {len(files)} event file(s) in {args.log_dir}")

    tags = [tag for tag in DEFAULT_TAGS if tag in scalars]
    if args.tag not in scalars:
        raise SystemExit(
            f"tag {args.tag!r} not found; available: {sorted(scalars)}"
        )
    if args.tag not in tags:
        tags.append(args.tag)

    epochs = sorted(scalars[args.tag])
    if args.last and len(epochs) > args.last:
        epochs = epochs[-args.last:]

    header = f"{'epoch':>6}  " + "".join(f"{tag:>15}" for tag in tags)
    print(header)
    print("-" * len(header))
    for step in epochs:
        row = f"{step + 1:>6}  "
        row += "".join(fmt(scalars[tag].get(step)) + "  " for tag in tags)
        print(row)

    series = scalars[args.tag]
    best_step = (min if args.rank_by == "min" else max)(series, key=series.get)
    checkpoints = match_checkpoints(args.save_dir, series.keys())
    print()
    print(f"[best] {args.tag}={series[best_step]:.6g} at epoch {best_step + 1}")
    if best_step + 1 in checkpoints:
        print(f"       {checkpoints[best_step + 1]}")
    else:
        print("       (no matching p-Ep*.pth found; check --save-dir)")

    if args.tag == "eval/mse_loss":
        tail = [series[step] for step in sorted(series)][-6:]
        if len(tail) >= 4:
            spread = (max(tail) - min(tail)) / max(abs(min(tail)), 1e-12)
            print()
            print(f"[hint] last {len(tail)} eval points span {spread * 100:.1f}% of the best value")
            print("       still improving -> keep going with --stage A (same-stage resume)")
            print("       plateaued      -> start --stage B from the checkpoint above")
            print("                         (--start-epoch 0 --reset-optim "
                  "--warm-start-res-head --lmbda 0.015 --diff-loss-scale auto)")
            print("       use --pick-stage-a to rank the plateau candidates properly")

    if args.pick_stage_a:
        chosen = pick_stage_a(scalars, args.plateau_tol)
        checkpoints = match_checkpoints(args.save_dir, scalars["eval/mse_loss"].keys())
        if chosen + 1 in checkpoints:
            print(f"[pick] {checkpoints[chosen + 1]}")
            print(
                "[pick] start Stage B with: --stage B --resume <that path> "
                "--start-epoch 0 --reset-optim --warm-start-res-head "
                "--lmbda 0.015 --diff-loss-scale auto"
            )
        else:
            print("[pick] (no matching p-Ep*.pth found; check --save-dir)")


if __name__ == "__main__":
    main()
