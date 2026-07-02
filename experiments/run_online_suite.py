"""
Run the complete online RL experiment suite across multiple seeds.

Each experiment writes to:
  results/online/<algo>/<preset>/seed_<N>/train_summary.json

Default experiment matrix
-------------------------
  sac / lcsac / lcsac_mean / lyap_rs_sac  ×  quadrotor_2d_track  (circle)
  sac / lcsac / lcsac_mean / lyap_rs_sac  ×  quadrotor_2d_stab
  sac / lcsac / lcsac_mean / lyap_rs_sac  ×  cartpole_stab
  sac / lcsac / lcsac_mean / lyap_rs_sac  ×  cartpole_track  (circle)

Default seeds: 1, 2, 3, 4, 5

EDMD prerequisite
-----------------
  Lyapunov algos (lcsac, lcsac_mean, lyap_rs_sac) require fitted EDMD models.
  If missing, those runs are skipped with instructions.

  Quadrotor 2D track:
      python -m edmd.collect_quad_2d && python -m edmd.train_quad_2d
  Quadrotor 2D stab:
      python -m edmd.collect_quad_2d_stab && python -m edmd.train_quad_2d_stab
  Cartpole:
      python -m edmd.collect_cartpole && python -m edmd.train_cartpole

Usage
-----
  python experiments/run_online_suite.py
  python experiments/run_online_suite.py --seeds 1
  python experiments/run_online_suite.py --algos sac
  python experiments/run_online_suite.py --presets quadrotor_2d_track cartpole_stab
  python experiments/run_online_suite.py --dry-run
  python experiments/run_online_suite.py --resume
  python experiments/run_online_suite.py --cpu

After completion:
  python experiments/compare_results.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_RL   = REPO_ROOT / "train_rl.py"
LOG_PATH   = REPO_ROOT / "results" / "online" / "suite_log.jsonl"

# (algo, preset, trajectory-or-None)
ONLINE_EXPERIMENTS = [
    ("sac",          "quadrotor_2d_track",  "circle"),
    ("sac",          "quadrotor_2d_stab",   None),
    ("lcsac",        "quadrotor_2d_track",  "circle"),
    ("lcsac",        "quadrotor_2d_stab",   None),
    ("lcsac_mean",    "quadrotor_2d_track",  "circle"),
    ("lcsac_mean",    "quadrotor_2d_stab",   None),
    ("lyap_rs_sac",  "quadrotor_2d_track",  "circle"),
    ("lyap_rs_sac",  "quadrotor_2d_stab",   None),
    ("sac",          "cartpole_stab",       None),
    ("sac",          "cartpole_track",      "circle"),
    ("lcsac",        "cartpole_stab",       None),
    ("lcsac",        "cartpole_track",      "circle"),
    ("lcsac_mean",    "cartpole_stab",       None),
    ("lcsac_mean",    "cartpole_track",      "circle"),
    ("lyap_rs_sac",  "cartpole_stab",       None),
    ("lyap_rs_sac",  "cartpole_track",      "circle"),
    ("sac",          "quadrotor_3d_track",  "circle"),
    ("sac",          "quadrotor_3d_stab",   None),
    ("lcsac",        "quadrotor_3d_track",  "circle"),
    ("lcsac",        "quadrotor_3d_stab",   None),
    ("lcsac_mean",   "quadrotor_3d_track",  "circle"),
    ("lcsac_mean",   "quadrotor_3d_stab",   None),
    ("lyap_rs_sac",  "quadrotor_3d_track",  "circle"),
    ("lyap_rs_sac",  "quadrotor_3d_stab",   None),
]

DEFAULT_SEEDS = [1, 2, 3, 4, 5]

# Algorithms that require EDMD assets (edmd_model + lqr_matrices).
EDMD_ALGOS = {"lcsac", "lcsac_mean", "lyap_rs_sac"}

# Per-preset EDMD assets required by EDMD_ALGOS.
# Points to tuned best-model artifacts (edmd.tune_* --retrain-best).
LCSAC_REQUIRED: dict[str, list[Path]] = {
    "quadrotor_2d_track": [
        REPO_ROOT / "results" / "edmd" / "quadrotor_2d_track" / "edmd_model.pkl",
        REPO_ROOT / "results" / "edmd" / "quadrotor_2d_track" / "lqr_matrices.npz",
    ],
    "quadrotor_2d_stab": [
        REPO_ROOT / "results" / "edmd" / "quadrotor_2d_stab" / "edmd_model.pkl",
        REPO_ROOT / "results" / "edmd" / "quadrotor_2d_stab" / "lqr_matrices.npz",
    ],
    "cartpole_stab": [
        REPO_ROOT / "results" / "edmd" / "cartpole" / "edmd_model.pkl",
        REPO_ROOT / "results" / "edmd" / "cartpole" / "lqr_matrices.npz",
    ],
    "cartpole_track": [
        REPO_ROOT / "results" / "edmd" / "cartpole" / "edmd_model.pkl",
        REPO_ROOT / "results" / "edmd" / "cartpole" / "lqr_matrices.npz",
    ],
    "quadrotor_3d_track": [
        REPO_ROOT / "results" / "edmd" / "quadrotor_3d" / "edmd_model.pkl",
        REPO_ROOT / "results" / "edmd" / "quadrotor_3d" / "lqr_matrices.npz",
    ],
    "quadrotor_3d_stab": [
        REPO_ROOT / "results" / "edmd" / "quadrotor_3d_stab" / "edmd_model.pkl",
        REPO_ROOT / "results" / "edmd" / "quadrotor_3d_stab" / "lqr_matrices.npz",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exp_output_dir(algo: str, preset: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "online" / algo / preset / f"seed_{seed}"


def is_complete(out: Path) -> bool:
    return (out / "train_summary.json").is_file()


def _run_one(algo, preset, trajectory, seed, cpu, dry_run) -> dict:
    out = exp_output_dir(algo, preset, seed)
    cmd = [
        sys.executable, str(TRAIN_RL),
        "--algo",       algo,
        "--preset",     preset,
        "--seed",       str(seed),
        "--output-dir", str(out),
        "--no-plots",
    ]
    if trajectory:
        cmd += ["--trajectory", trajectory]
    if cpu:
        cmd.append("--cpu")

    entry = {
        "algo": algo, "preset": preset, "seed": seed,
        "trajectory": trajectory, "output_dir": str(out),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(str(x) for x in cmd)}")
        entry.update(status="dry_run", exit_code=None, duration_s=None)
        return entry

    out.mkdir(parents=True, exist_ok=True)
    print(f"  CMD: {' '.join(str(x) for x in cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    dt = round(time.time() - t0, 1)
    entry.update(exit_code=rc, duration_s=dt,
                 status="done" if rc == 0 else "failed")
    print(f"  → exit {rc} in {dt}s")
    return entry


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _print_summary(experiments, seeds) -> None:
    """After all runs, print best_eval_reward from each train_summary.json."""
    print(f"\n{'='*80}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'algo':<10} {'preset':<28} {'seed':>4}  {'best_reward':>12}")
    print(f"  {'-'*10} {'-'*28} {'-'*4}  {'-'*12}")
    for algo, preset, _ in experiments:
        for seed in seeds:
            out = exp_output_dir(algo, preset, seed)
            summary_path = out / "train_summary.json"
            if summary_path.is_file():
                with open(summary_path, encoding="utf-8") as f:
                    s = json.load(f)
                reward = s.get("best_eval_reward", "N/A")
                reward_str = f"{reward:.4f}" if isinstance(reward, float) else str(reward)
            else:
                reward_str = "missing"
            print(f"  {algo:<10} {preset:<28} {seed:>4}  {reward_str:>12}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help=f"Seeds to run (default: {DEFAULT_SEEDS})")
    p.add_argument("--algos", nargs="+", default=None,
                   help="Filter by algo, e.g. --algos sac lcsac")
    p.add_argument("--presets", nargs="+", default=None,
                   help="Filter by preset, e.g. --presets quadrotor_2d_track")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running")
    p.add_argument("--resume", action="store_true",
                   help="Skip experiments that already have train_summary.json")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even when CUDA is available")
    args = p.parse_args()

    exps = list(ONLINE_EXPERIMENTS)
    if args.algos:
        exps = [(a, pr, t) for a, pr, t in exps if a in args.algos]
    if args.presets:
        exps = [(a, pr, t) for a, pr, t in exps if pr in args.presets]

    total = len(exps) * len(args.seeds)
    print(f"Online suite: {len(exps)} experiments × {len(args.seeds)} seeds = {total} runs")
    print(f"Seeds : {args.seeds}")
    if args.resume:
        print("Resume: skipping completed runs")
    if args.dry_run:
        print("Dry-run: commands will be printed but not executed")

    # Per-preset LC-SAC asset check
    lcsac_preset_ok: dict[str, bool] = {
        p: all(f.is_file() for f in files)
        for p, files in LCSAC_REQUIRED.items()
    }
    for preset, files in LCSAC_REQUIRED.items():
        needs_edmd = any(a in EDMD_ALGOS and pr == preset for a, pr, _ in exps)
        if not lcsac_preset_ok[preset] and needs_edmd:
            print(f"\nWARNING: EDMD assets missing for {preset}:")
            for f in files:
                tag = "OK" if f.is_file() else "MISSING"
                print(f"  [{tag}] {f.relative_to(REPO_ROOT)}")
            if preset == "quadrotor_2d_stab":
                print("  To prepare:")
                print("    python -m edmd.collect_quad_2d_stab")
                print("    python -m edmd.train_quad_2d_stab")
            elif preset in ("cartpole_stab", "cartpole_track"):
                print("  To prepare:")
                print("    python -m edmd.collect_cartpole")
                print("    python -m edmd.train_cartpole")
            else:
                print("  To prepare:")
                print("    python -m edmd.collect_quad_2d")
                print("    python -m edmd.train_quad_2d")

    done_n = fail_n = skip_n = 0
    run_idx = 0

    for algo, preset, traj in exps:
        if algo in EDMD_ALGOS and not lcsac_preset_ok.get(preset, False):
            print(f"\n[SKIP] {algo} / {preset} — missing EDMD assets")
            skip_n += len(args.seeds)
            continue

        for seed in args.seeds:
            run_idx += 1
            out  = exp_output_dir(algo, preset, seed)
            label = f"{algo} / {preset} / seed_{seed}"
            print(f"\n[{run_idx}/{total}] {label}")

            if args.resume and is_complete(out):
                print("  [SKIP] train_summary.json already present")
                skip_n += 1
                continue

            entry = _run_one(algo, preset, traj, seed, args.cpu, args.dry_run)
            _append_log(entry)

            if entry["status"] == "done":
                done_n += 1
            elif entry["status"] == "failed":
                fail_n += 1
            else:
                skip_n += 1

    # Final summary
    print(f"\n{'='*50}")
    print(f"Online suite finished: {done_n} done, {fail_n} failed, {skip_n} skipped")
    if not args.dry_run:
        print(f"Log: {LOG_PATH}")
        _print_summary(exps, args.seeds)

    return 1 if fail_n else 0


if __name__ == "__main__":
    raise SystemExit(main())
