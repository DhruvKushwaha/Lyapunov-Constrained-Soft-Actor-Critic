"""
Compare training results across all algorithms and environments.

Reads experiment outputs from ``results/online/<algo>/<preset>/seed_<N>/``
and generates:

  1. Bar chart — best eval reward per algo × preset (mean ± std across seeds)
  2. Learning curves — eval reward vs training steps per preset
  3. Lyapunov loss — LC-SAC / LC-SAC-Mean (mean ± std across seeds)
  4. Summary CSV + JSON

Data sources per run directory
-------------------------------
  train_summary.json          always present (best_eval_reward)
  eval_rewards<suffix>.json   [{step, reward}, ...]   (written by rl/train.py)
  episode_rewards<suffix>.npy 1-D array of per-episode totals
  lyap_loss<suffix>.json      LC-SAC / LC-SAC-Mean / Lyap-RS-SAC: [{step, lyap_loss, ...}, ...]

Usage
-----
  python experiments/compare_results.py
  python experiments/compare_results.py --exp-dir results/online
  python experiments/compare_results.py --preset quadrotor_2d_track
  python experiments/compare_results.py --output-dir my_plots/
  python experiments/compare_results.py --seeds 1 2 3
  python experiments/compare_results.py --no-show        # save without plt.show()
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_ALGOS = ["sac", "lcsac", "lcsac_mean", "lyap_rs_sac"]

# Algorithms that emit lyap_loss*.json during training
LYAP_LOSS_ALGOS = {"lcsac", "lcsac_mean", "lyap_rs_sac"}

# Color palette — consistent across all plots
ALGO_COLORS = {
    "sac":         "C0",
    "lcsac":       "C1",
    "lcsac_mean":   "C2",
    "lyap_rs_sac": "C3",
}
ALGO_LABELS = {
    "sac":         "SAC",
    "lcsac":       "LC-SAC",
    "lcsac_mean":   "LC-SAC-Mean",
    "lyap_rs_sac": "Lyap-RS-SAC",
}
PRESET_LABELS = {
    "quadrotor_2d_track":    "Quad-2D Track",
    "quadrotor_2d_stab":     "Quad-2D Stab",
    "quadrotor_3d_track":    "Quad-3D Track",
    "quadrotor_3d_track_gym":"Quad-3D Track (Gym)",
    "quadrotor_3d_stab":     "Quad-3D Stab",
    "cartpole_stab":         "Cartpole Stab",
    "cartpole_track":        "Cartpole Track",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _primary_metric(summary: dict) -> float | None:
    """Extract the main scalar performance metric from a train_summary.json."""
    v = summary.get("best_eval_reward")
    return float(v) if v is not None else None


def _load_eval_rewards(run_dir: Path) -> list[dict] | None:
    """Load [{step, reward}, ...] from eval_rewards*.json, returning first match."""
    for p in sorted(run_dir.glob("eval_rewards*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if data and isinstance(data[0], dict) and "step" in data[0]:
                return data
        except Exception:
            pass
    return None



def _load_lyap_loss(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (steps, losses) from lyap_loss*.json."""
    for p in sorted(run_dir.glob("lyap_loss*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            steps  = np.array([e["step"]      for e in data])
            losses = np.array([e["lyap_loss"] for e in data], dtype=float)
            if len(steps) > 0:
                return steps, losses
        except Exception:
            pass
    return None


def scan_experiments(exp_dir: Path, seeds: list[int] | None) -> dict:
    """
    Scan exp_dir for completed runs.

    Returns nested dict: result[preset][algo] = list of per-seed dicts:
      {seed, summary, eval_curve, lyap_curve}
    """
    result: dict[str, dict[str, list]] = {}

    for summary_path in sorted(exp_dir.rglob("train_summary.json")):
        run_dir = summary_path.parent
        parts   = run_dir.relative_to(exp_dir).parts
        if len(parts) != 3:          # algo / preset / seed_N
            continue
        algo, preset, seed_dir = parts
        if not seed_dir.startswith("seed_"):
            continue
        try:
            seed = int(seed_dir.split("_", 1)[1])
        except ValueError:
            continue
        if seeds and seed not in seeds:
            continue

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        eval_curve = _load_eval_rewards(run_dir)
        lyap_curve = _load_lyap_loss(run_dir) if algo in LYAP_LOSS_ALGOS else None

        result.setdefault(preset, {}).setdefault(algo, []).append({
            "seed":       seed,
            "summary":    summary,
            "metric":     _primary_metric(summary),
            "eval_curve": eval_curve,
            "lyap_curve": lyap_curve,
        })

    # Sort seeds within each group
    for preset_data in result.values():
        for runs in preset_data.values():
            runs.sort(key=lambda r: r["seed"])

    return result


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _interp_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate (x, y) curves to a common x grid; return (x_grid, mean, std)."""
    if not curves:
        return None
    all_x = np.concatenate([x for x, _ in curves])
    x_grid = np.linspace(all_x.min(), all_x.max(), 500)
    ys = []
    for x, y in curves:
        if len(x) < 2:
            continue
        ys.append(np.interp(x_grid, x, y))
    if not ys:
        return None
    ys = np.array(ys)
    return x_grid, ys.mean(axis=0), ys.std(axis=0)


# ---------------------------------------------------------------------------
# Plot 1: Bar chart of best eval reward per preset
# ---------------------------------------------------------------------------

def plot_bar_comparison(data: dict, output_dir: Path, show: bool) -> None:
    presets = sorted(data.keys())
    if not presets:
        return

    n_presets = len(presets)
    fig, axes = plt.subplots(
        1, n_presets,
        figsize=(max(6, 4 * n_presets), 6),
        squeeze=False,
    )

    for col, preset in enumerate(presets):
        ax = axes[0][col]
        algo_data = data[preset]
        present_algos = [a for a in ALL_ALGOS if a in algo_data]

        means, stds, colors, labels = [], [], [], []
        for algo in present_algos:
            metrics = [r["metric"] for r in algo_data[algo] if r["metric"] is not None]
            if not metrics:
                continue
            means.append(float(np.mean(metrics)))
            stds.append(float(np.std(metrics)))
            colors.append(ALGO_COLORS[algo])
            labels.append(ALGO_LABELS[algo])

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                      alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(PRESET_LABELS.get(preset, preset), fontsize=11, fontweight="bold")
        ax.set_ylabel("Best Eval Reward" if col == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Annotate bar tops
        for bar, m, s in zip(bars, means, stds):
            ax.annotate(
                f"{m:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01 * abs(m)),
                ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Best Eval Reward — All Algorithms × Environments\n(mean ± std across seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_dir / "bar_best_eval_reward.png", show)


# ---------------------------------------------------------------------------
# Plot 2: Learning curves per preset
# ---------------------------------------------------------------------------

def plot_learning_curves(data: dict, output_dir: Path, show: bool) -> None:
    for preset, algo_data in sorted(data.items()):
        has_curve = {}
        for algo, runs in algo_data.items():
            curves = [(np.array([e["step"] for e in r["eval_curve"]]),
                       np.array([e["reward"] for e in r["eval_curve"]]))
                      for r in runs if r.get("eval_curve")]
            if curves:
                has_curve[algo] = curves

        if not has_curve:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        for algo, curves in sorted(has_curve.items()):
            result = _interp_curves(curves)
            if result is None:
                continue
            x_grid, mean_y, std_y = result
            color = ALGO_COLORS.get(algo, "gray")
            lbl   = ALGO_LABELS.get(algo, algo)
            ax.plot(x_grid, mean_y, linewidth=2, color=color, label=lbl, zorder=3)
            ax.fill_between(x_grid, mean_y - std_y, mean_y + std_y,
                            alpha=0.2, color=color, zorder=2)
            for x, y in curves:
                ax.plot(x, y, linewidth=0.6, alpha=0.25, color=color, zorder=1)

        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Eval Reward", fontsize=11)
        ax.set_title(
            f"Learning Curves — {PRESET_LABELS.get(preset, preset)}\n(mean ± std across seeds)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, output_dir / f"learning_curves_{preset}.png", show)


# ---------------------------------------------------------------------------
# Plot 3: Lyapunov loss (LC-SAC only)
# ---------------------------------------------------------------------------

def plot_lyapunov_loss(data: dict, output_dir: Path, show: bool) -> None:
    for preset, algo_data in sorted(data.items()):
        present = [a for a in LYAP_LOSS_ALGOS if a in algo_data]
        if not present:
            continue

        fig, ax = plt.subplots(figsize=(9, 4))
        plotted = False

        for algo in sorted(present):
            curves = [r["lyap_curve"] for r in algo_data[algo] if r.get("lyap_curve")]
            if not curves:
                continue
            result = _interp_curves(curves)
            if result is None:
                continue
            x_grid, mean_y, std_y = result
            color = ALGO_COLORS[algo]
            lbl   = ALGO_LABELS[algo]

            for steps, losses in curves:
                ax.semilogy(steps, losses, linewidth=0.7, alpha=0.25, color=color, zorder=1)
            ax.semilogy(x_grid, mean_y, linewidth=2.5, color=color,
                        label=f"{lbl} (mean)", zorder=3)
            ax.fill_between(x_grid,
                            np.maximum(mean_y - std_y, 1e-10),
                            mean_y + std_y,
                            alpha=0.2, color=color, zorder=2)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Lyapunov Loss (log scale)", fontsize=11)
        ax.set_title(f"Lyapunov Loss — {PRESET_LABELS.get(preset, preset)}\n(mean ± std across seeds)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        _save_fig(fig, output_dir / f"lyapunov_loss_{preset}.png", show)


# ---------------------------------------------------------------------------
# Plot 4: Summary table (text print + CSV + JSON)
# ---------------------------------------------------------------------------

def write_summary(data: dict, output_dir: Path) -> None:
    rows = []
    for preset, algo_data in sorted(data.items()):
        for algo in ALL_ALGOS:
            if algo not in algo_data:
                continue
            for run in algo_data[algo]:
                rows.append({
                    "preset":      preset,
                    "algo":        algo,
                    "seed":        run["seed"],
                    "metric":      run["metric"],
                })

    # Print table
    print(f"\n{'='*72}")
    print(" RESULTS SUMMARY")
    print(f"{'='*72}")
    hdr = f"  {'preset':<28} {'algo':<8} {'seed':>4}  {'metric':>12}"
    print(hdr)
    print(f"  {'-'*28} {'-'*8} {'-'*4}  {'-'*12}")
    for r in rows:
        m = f"{r['metric']:.4f}" if r["metric"] is not None else "N/A"
        print(f"  {r['preset']:<28} {r['algo']:<8} {r['seed']:>4}  {m:>12}")

    # Aggregate: mean ± std per (preset, algo)
    print(f"\n{'='*72}")
    print(" AGGREGATED  (mean ± std across seeds)")
    print(f"{'='*72}")
    print(f"  {'preset':<28} {'algo':<8}  {'mean':>10}  {'std':>8}  {'n':>3}")
    print(f"  {'-'*28} {'-'*8}  {'-'*10}  {'-'*8}  {'-'*3}")
    agg_rows = []
    for preset, algo_data in sorted(data.items()):
        for algo in ALL_ALGOS:
            if algo not in algo_data:
                continue
            metrics = [r["metric"] for r in algo_data[algo] if r["metric"] is not None]
            if not metrics:
                continue
            mean_v = float(np.mean(metrics))
            std_v  = float(np.std(metrics))
            n      = len(metrics)
            print(f"  {preset:<28} {algo:<8}  {mean_v:>10.4f}  {std_v:>8.4f}  {n:>3}")
            agg_rows.append({"preset": preset, "algo": algo,
                             "mean": mean_v, "std": std_v, "n": n})

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["preset", "algo", "seed", "metric"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved per-seed summary to {csv_path}")

    agg_path = output_dir / "summary_aggregated.csv"
    if agg_rows:
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["preset", "algo", "mean", "std", "n"])
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"Saved aggregated summary to {agg_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--exp-dir", default=None,
                   help="Root of experiment outputs (default: results/online/)")
    p.add_argument("--output-dir", default=None,
                   help="Where to save plots (default: <exp-dir>/plots/)")
    p.add_argument("--preset", nargs="+", default=None,
                   help="Filter to specific presets")
    p.add_argument("--algos", nargs="+", default=None,
                   choices=ALL_ALGOS,
                   help="Filter to specific algorithms (e.g. --algos sac lcsac lcsac_mean lyap_rs_sac)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Filter to specific seeds")
    p.add_argument("--no-show", action="store_true",
                   help="Save figures without calling plt.show()")
    args = p.parse_args()

    exp_dir    = Path(args.exp_dir) if args.exp_dir else REPO_ROOT / "results" / "online"
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "plots"
    show       = not args.no_show

    if not exp_dir.is_dir():
        print(f"Experiment directory not found: {exp_dir}")
        return 2

    print(f"Scanning {exp_dir} …")
    data = scan_experiments(exp_dir, args.seeds)

    if args.preset:
        data = {k: v for k, v in data.items() if k in args.preset}
    if args.algos:
        data = {preset: {a: runs for a, runs in algo_data.items() if a in args.algos}
                for preset, algo_data in data.items()}
        data = {k: v for k, v in data.items() if v}

    if not data:
        print("No completed runs found.")
        return 1

    # Count runs
    total = sum(len(runs) for ad in data.values() for runs in ad.values())
    print(f"Found {total} completed runs across {len(data)} presets.")

    write_summary(data, output_dir)

    print("\nGenerating bar chart …")
    plot_bar_comparison(data, output_dir, show)

    print("Generating learning curves …")
    plot_learning_curves(data, output_dir, show)

    print("Generating Lyapunov loss plots …")
    plot_lyapunov_loss(data, output_dir, show)

    print(f"\nAll plots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
