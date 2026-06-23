import argparse
import os
import re
import subprocess
import sys


def safe_log_name(model_path):
    name = model_path.strip()
    name = re.sub(r"^[.][\\/]+", "", name)
    name = re.sub(r"[\\/]+", "_", name)
    name = re.sub(r"[:*?\"<>|]", "_", name)
    name = name.strip("._ ")
    if not name:
        name = "model"
    return "log-{}.txt".format(name)


def main():
    parser = argparse.ArgumentParser(
        description="Run test4-eval-diff-sit.py for multiple P-frame models and tau values."
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="P-frame checkpoint paths.")
    parser.add_argument("--taus", nargs="+", type=int, required=True,
                        help="Tau values to test.")
    parser.add_argument("--context-type", type=str, default="simple",
                        choices=["legacy", "simple", "diffusion_fusion"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--data-path", type=str, default="../Mosmed/test-full")
    parser.add_argument("--i-frame-ckp", type=str, default="ckp_nll/0001/ckp.tar")
    parser.add_argument("--log-dir", type=str, default=".")
    parser.add_argument("--test-script", type=str, default="test4-eval-diff-sit.py")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    for model in args.models:
        for tau in args.taus:
            log_name = safe_log_name(model)
            if len(args.taus) > 1:
                root, ext = os.path.splitext(log_name)
                log_name = "{}-tau{}{}".format(root, tau, ext)
            log_file = os.path.join(args.log_dir, log_name)

            cmd = [
                args.python,
                args.test_script,
                "--tau", str(tau),
                "--steps", str(args.steps),
                "--context-type", args.context_type,
                "--data-path", args.data_path,
                "--i-frame-ckp", args.i_frame_ckp,
                "--p-frame-ckp", model,
                "--log-file", log_file,
            ]

            print("\n[batch] model={} tau={} log={}".format(model, tau, log_file))
            print("[batch] {}".format(" ".join(cmd)))
            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
