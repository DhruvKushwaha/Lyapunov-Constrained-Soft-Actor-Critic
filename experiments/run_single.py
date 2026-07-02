"""
Run a single training experiment (one algo × preset × seed).

Output is written to results/online/<algo>/<preset>/seed_<N>/
unless overridden with --output-dir.

Examples
--------
  python experiments/run_single.py --algo sac  --preset quadrotor_2d_track --seed 1
  python experiments/run_single.py --algo sac  --preset quadrotor_2d_track --seed 1 --trajectory figure8
  python experiments/run_single.py --algo lcsac --preset quadrotor_2d_track --seed 2
  python experiments/run_single.py --algo sac  --preset quadrotor_3d_track --seed 1
  python experiments/run_single.py --algo sac  --preset cartpole_stab --seed 3

  # Offline baselines (dataset must exist — see experiments/run_offline_suite.py)
  python experiments/run_single.py --algo cql --preset quadrotor_2d_track --seed 1 \
      --dataset Saved_data/offline_pid_2d_track.npz
  python experiments/run_single.py --algo td3bc --preset quadrotor_2d_track --seed 2 \
      --dataset Saved_data/offline_pid_2d_track.npz

  # Collect random transitions on the fly (no existing dataset needed)
  python experiments/run_single.py --algo cql --preset quadrotor_2d_track --seed 1 \
      --collect-random 100000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_RL   = REPO_ROOT / "train_rl.py"
OFFLINE_ALGOS = {"cql", "td3bc", "mopo", "combo"}

ALL_PRESETS = [
    "quadrotor_2d_track",
    "quadrotor_2d_stab",
    "quadrotor_3d_track",
    "quadrotor_3d_track_gym",
    "quadrotor_3d_stab",
    "cartpole_stab",
]


def default_output_dir(algo: str, preset: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "online" / algo / preset / f"seed_{seed}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--algo", required=True,
                   choices=("sac", "lcsac", "cql", "td3bc", "mopo", "combo"))
    p.add_argument("--preset", required=True, choices=ALL_PRESETS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trajectory", default=None, choices=("circle", "figure8", "square"),
                   help="Trajectory type for quadrotor traj_tracking presets")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default: results/online/<algo>/<preset>/seed_<N>)")
    p.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")
    p.add_argument("--plots", action="store_true", default=False,
                   help="Enable matplotlib training plots (disabled by default for automation)")
    # Offline-specific
    p.add_argument("--dataset", default=None,
                   help="(offline algos) Path to .npz MDP-transition dataset")
    p.add_argument("--collect-random", type=int, default=None, metavar="N",
                   help="(offline algos) Collect N random-action transitions if --dataset not set")
    p.add_argument("--offline-epoch", type=int, default=300)
    p.add_argument("--offline-step-per-epoch", type=int, default=1000)
    p.add_argument("--offline-batch-size", type=int, default=256)
    args = p.parse_args()

    out = (
        Path(args.output_dir) if args.output_dir
        else default_output_dir(args.algo, args.preset, args.seed)
    )
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAIN_RL),
        "--algo",       args.algo,
        "--preset",     args.preset,
        "--seed",       str(args.seed),
        "--output-dir", str(out),
    ]
    if not args.plots:
        cmd.append("--no-plots")
    if args.trajectory:
        cmd += ["--trajectory", args.trajectory]
    if args.cpu:
        cmd.append("--cpu")
    if args.algo in OFFLINE_ALGOS:
        if args.dataset:
            cmd += ["--offline-dataset", args.dataset]
        elif args.collect_random:
            cmd += ["--collect-random-transitions", str(args.collect_random)]
        cmd += [
            "--offline-epoch",          str(args.offline_epoch),
            "--offline-step-per-epoch", str(args.offline_step_per_epoch),
            "--offline-batch-size",     str(args.offline_batch_size),
        ]

    print(f"Output : {out}")
    print(f"Command: {' '.join(str(x) for x in cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    dt = time.time() - t0
    status = "SUCCESS" if rc == 0 else "FAILED"
    print(f"[{status}] in {dt:.1f}s (exit code {rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
