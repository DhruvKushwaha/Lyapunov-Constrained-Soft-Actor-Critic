"""
Grid-search EDMD + LQR hyperparameters for cartpole tracking-error dynamics.

Two-phase search:
  Phase 1 — EDMD phase: train one Koopman model per (n_rbf_centers, rbf_width, regularization)
             combo, compute one-step prediction RMSE.
  Phase 2 — LQR phase:  for each EDMD model, sweep (q_x, q_phi) cost weights and solve DARE.
             Valid configs satisfy hard control-theory criteria; all valid configs are ranked.

Hard criteria (must pass to be shown):
  - controllability_full  = True
  - A_cl_spectral_radius  < 1.0         (closed-loop stable)
  - P_min_eig_real        > 0.0         (P positive definite)
  - P_num_negative_eig    = 0

Ranking (PRIMARY = P matrix quality, most important for Lyapunov RL):
  1. P_condition_number   (lower = better conditioned V(z) = z^T P z)
  2. P_min_eig_real       (higher = further from singularity)
  3. A_cl_spectral_radius (lower = larger LQR convergence margin)
  4. one_step_rmse_mean   (lower = better Koopman prediction)

Configs with P_condition_number ≤ p_cond_max are marked "✓"; others "~" but still shown.
Cartpole note: A_cl ρ = 0.9978 baseline means any config with A_cl ρ < 0.99 is a clear win
for Lyapunov constraint satisfiability (the constraint requires ~1% V decrease per step).

Cartpole specifics:
  - State dimension:   4  (x, x_dot, theta, theta_dot)
  - Action dimension:  1  (scalar force)
  - Control frequency: 15 Hz  (dt = 1/15)
  - Lifted dim:        4 + n_rbf_centers

Usage
-----
  python -m edmd.tune_cartpole --quick
  python -m edmd.tune_cartpole
  python -m edmd.tune_cartpole --quick --retrain-best
  python -m edmd.tune_cartpole --data results/edmd/data/cartpole.npz --output results/edmd/cartpole
"""

import argparse
import json
import pickle
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_are
from sklearn.cluster import KMeans

import pykoopman as pk
from pykoopman.regression import EDMDc

from edmd.metrics import collect_matrix_metrics, one_step_metrics

# ---------------------------------------------------------------------------
# Search grids
# ---------------------------------------------------------------------------
FULL_GRID = {
    "n_rbf_centers": [3, 5, 8, 12, 16],
    "rbf_width":     [0.1, 0.25, 0.35, 0.5, 1.0],
    "regularization": [1e-6, 1e-5, 1e-4],
    "q_x":           [0.1, 1.0, 10.0],
    # Extended q_phi range: higher values improve P conditioning at cost of lifting semantics
    "q_phi":         [1e-6, 1e-5, 1e-4, 1e-3],
}

QUICK_GRID = {
    "n_rbf_centers": [3, 5, 8],
    "rbf_width":     [0.1, 0.35, 1.0],
    "regularization": [1e-5, 1e-4],
    "q_x":           [1.0],
    "q_phi":         [1e-6, 1e-4, 1e-3],
}

DT = 1 / 15.0
TOP_N = 15


# ---------------------------------------------------------------------------
# EDMD training
# ---------------------------------------------------------------------------
def _train_edmd(X, X_next, U, n_rbf_centers, rbf_width, regularization):
    kmeans = KMeans(n_clusters=n_rbf_centers, random_state=42, n_init=12)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.T

    rbf = pk.observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=centers.shape[1],
        centers=centers,
        kernel_width=rbf_width,
        polyharmonic_coeff=1.0,
        include_state=True,
    )
    model = pk.Koopman(observables=rbf, regressor=EDMDc(), quiet=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y=X_next, u=U, dt=DT)
    return model


# ---------------------------------------------------------------------------
# LQR / DARE sweep
# ---------------------------------------------------------------------------
def _try_lqr(A, B, n_error, q_x, q_phi):
    lifted_dim = A.shape[0]
    m = lifted_dim - n_error
    Q = np.diag([q_x] * n_error + [q_phi] * m)
    R = np.eye(B.shape[1])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = solve_discrete_are(A, B, Q, R)
        P = (P + P.T) / 2
    except (LinAlgError, ValueError):
        return None, False

    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    A_cl = A - B @ K

    metrics = collect_matrix_metrics(A, B, P, K, A_cl, n_error)

    eig_real = np.real(np.linalg.eigvals(P))
    p_min = float(np.min(eig_real))
    p_max = float(np.max(eig_real))
    p_cond = p_max / (p_min + 1e-12) if p_min > 0 else float("inf")
    metrics["P_condition_number"] = p_cond

    # Hard criteria: physical requirements only (P>0, A_cl<1, controllable)
    passed = (
        metrics["controllability_full"]
        and metrics["A_cl_spectral_radius"] < 1.0
        and metrics["P_min_eig_real"] > 0.0
        and metrics["P_num_negative_eig_real"] == 0
    )
    return metrics, passed


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------
def run_search(X, X_next, U, grid, verbose):
    n_error = X.shape[1]
    edmd_keys = ["n_rbf_centers", "rbf_width", "regularization"]
    lqr_keys  = ["q_x", "q_phi"]

    edmd_combos = list(product(*[grid[k] for k in edmd_keys]))
    lqr_combos  = list(product(*[grid[k] for k in lqr_keys]))

    n_edmd = len(edmd_combos)
    n_lqr  = len(lqr_combos)
    print(f"Search space: {n_edmd} EDMD configs × {n_lqr} LQR configs = {n_edmd * n_lqr} combos total")

    valid = []
    invalid = []
    edmd_failed = 0

    for ei, (n_rbf, width, reg) in enumerate(edmd_combos):
        if verbose:
            print(f"\n[{ei+1}/{n_edmd}] n_rbf={n_rbf}, width={width}, reg={reg:.0e} — training …",
                  end="  ", flush=True)

        try:
            model = _train_edmd(X, X_next, U, n_rbf, width, reg)
        except Exception as exc:
            if verbose:
                print(f"EDMD failed: {exc}")
            edmd_failed += 1
            continue

        A = model.A
        B = model.B
        os_m = one_step_metrics(model, X, X_next, U)
        rmse_mean = float(np.mean(os_m["rmse_per_component"]))
        if verbose:
            print(f"RMSE={rmse_mean:.4f}", end="  ", flush=True)

        n_valid_lqr = 0
        for q_x, q_phi in lqr_combos:
            metrics, passed = _try_lqr(A, B, n_error, q_x, q_phi)
            if metrics is None:
                continue
            entry = {
                "n_rbf_centers": int(n_rbf),
                "rbf_width":     float(width),
                "regularization": float(reg),
                "q_x":   float(q_x),
                "q_phi": float(q_phi),
                "one_step_rmse_mean": rmse_mean,
                **{k: v for k, v in metrics.items()},
            }
            if passed:
                valid.append(entry)
                n_valid_lqr += 1
            else:
                invalid.append(entry)

        if verbose:
            print(f"→ {n_valid_lqr}/{n_lqr} valid")

    print(f"\nSearch complete: {len(valid)} valid / {len(valid)+len(invalid)} attempted "
          f"({edmd_failed} EDMD failures skipped)")
    return valid, invalid


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _sort_key(entry):
    """Primary: P_condition_number (lower = better conditioned Lyapunov fn — most important).
    Secondary: P_min_eig_real (higher = further from singularity, negate for ascending sort).
    Tertiary: A_cl_spectral_radius (lower = larger LQR convergence margin).
    Quaternary: one-step RMSE mean (lower = better Koopman prediction)."""
    return (
        entry.get("P_condition_number", float("inf")),
        -entry.get("P_min_eig_real", 0.0),
        entry["A_cl_spectral_radius"],
        entry["one_step_rmse_mean"],
    )


def print_table(results, title, top_n=TOP_N, p_cond_max=float("inf")):
    import math
    ranked = sorted(results, key=_sort_key)[:top_n]
    print(f"\n{'='*120}")
    print(f" {title}  (top {len(ranked)} shown, sorted by P conditioning)")
    print(f"{'='*120}")
    hdr = (
        f"{'#':>3}  {'n_rbf':>5}  {'width':>6}  {'reg':>7}  "
        f"{'q_x':>5}  {'q_phi':>7}  "
        f"{'log10(cond)':>11}  {'P_min_eig':>10}  {'P_max_eig':>10}  "
        f"{'A_cl_sr':>8}  {'RMSE':>7}  {'ok':>3}"
    )
    print(hdr)
    print("-" * 120)
    for i, e in enumerate(ranked, 1):
        p_cond = e.get("P_condition_number", float("nan"))
        log_cond = math.log10(p_cond) if (p_cond > 0 and math.isfinite(p_cond)) else float("nan")
        ok = "✓" if p_cond <= p_cond_max else "~"
        print(
            f"{i:>3}  {e['n_rbf_centers']:>5}  {e['rbf_width']:>6.2f}  "
            f"{e['regularization']:>7.0e}  {e['q_x']:>5.1f}  {e['q_phi']:>7.0e}  "
            f"{log_cond:>11.2f}  {e['P_min_eig_real']:>10.3e}  {e['P_max_eig_real']:>10.3e}  "
            f"{e['A_cl_spectral_radius']:>8.4f}  {e['one_step_rmse_mean']:>7.4f}  {ok:>3}"
        )
    return ranked


# ---------------------------------------------------------------------------
# Re-train best config and save artifacts
# ---------------------------------------------------------------------------
def retrain_and_save(best, X, X_next, U, output_dir: Path):
    print(f"\n{'='*60}")
    print(" Re-training best config and saving artifacts …")
    print(f"{'='*60}")
    n_rbf  = best["n_rbf_centers"]
    width  = best["rbf_width"]
    reg    = best["regularization"]
    q_x    = best["q_x"]
    q_phi  = best["q_phi"]
    n_error = X.shape[1]

    print(f"  n_rbf={n_rbf}, width={width}, reg={reg:.0e}, q_x={q_x}, q_phi={q_phi:.0e}")

    model = _train_edmd(X, X_next, U, n_rbf, width, reg)
    A = model.A
    B = model.B
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho_A > 1.0:
        A = A / rho_A
        print(f"  A normalised: raw rho={rho_A:.4f} -> scaled rho=1.000")
    lifted_dim = A.shape[0]
    m = lifted_dim - n_error
    Q = np.diag([q_x] * n_error + [q_phi] * m)
    R = np.eye(B.shape[1])
    P = solve_discrete_are(A, B, Q, R)
    P = (P + P.T) / 2
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    A_cl = A - B @ K

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path   = output_dir / "edmd_model.pkl"
    matrices_path = output_dir / "lqr_matrices.npz"
    config_path  = output_dir / "best_config.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    np.savez(matrices_path, A_lifted=A, B_lifted=B, A_cl=A_cl, P=P, K=K, Q=Q, R=R)
    with open(config_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"  Saved model     → {model_path}")
    print(f"  Saved matrices  → {matrices_path}")
    print(f"  Saved config    → {config_path}")
    print(f"\n  A_cl spectral radius : {np.max(np.abs(np.linalg.eigvals(A_cl))):.6f}")
    print(f"  P min eigenvalue     : {np.min(np.real(np.linalg.eigvals(P))):.3e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="Use coarse grid (~72 combos) instead of full grid (~1125)")
    parser.add_argument("--retrain-best", action="store_true",
                        help="Re-train and save artifacts for the best passing config")
    parser.add_argument("--p-cond-max", type=float, default=1e9,
                        help="P condition number threshold for the ✓ marker in the table "
                             "(default: 1e9). All valid configs are shown regardless.")
    parser.add_argument("--data", type=str, default="results/edmd/data/cartpole.npz",
                        help="Path to .npz data file")
    parser.add_argument("--output", type=str, default="results/edmd/cartpole",
                        help="Output directory for results JSON and best-config artifacts")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-EDMD-config progress")
    args = parser.parse_args()

    data_file = Path(args.data)
    output_dir = Path(args.output)

    if not data_file.is_file():
        print(
            f"Data file not found: {data_file}\n"
            "Run: python -m edmd.collect_cartpole\nthen retry.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading data from {data_file} …")
    raw = np.load(data_file)
    X      = np.asarray(raw["tracking_error"],      dtype=np.float64)
    X_next = np.asarray(raw["tracking_error_next"], dtype=np.float64)
    U      = np.asarray(raw["U"],                   dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    print(f"  X={X.shape}, X_next={X_next.shape}, U={U.shape}")

    grid = QUICK_GRID if args.quick else FULL_GRID
    mode = "QUICK" if args.quick else "FULL"
    print(f"\nUsing {mode} grid.  P cond annotation threshold: {args.p_cond_max:.0e}")

    valid, invalid = run_search(X, X_next, U, grid, args.verbose)

    if not valid:
        print(
            "\nNo valid configs found (all failed hard criteria: P>0, A_cl<1, controllable).\n"
            "Suggestions:\n"
            "  • Try larger n_rbf_centers\n"
            "  • Try smaller rbf_width\n"
            "  • Check data quality: run edmd.collect_cartpole with more episodes"
        )
        return 1

    import math
    ranked = print_table(valid, "ALL VALID CONFIGS (ranked by P conditioning)", top_n=TOP_N,
                         p_cond_max=args.p_cond_max)
    best = ranked[0]

    p_cond_best = best.get("P_condition_number", float("nan"))
    log_cond = (math.log10(p_cond_best)
                if math.isfinite(p_cond_best) and p_cond_best > 0 else float("nan"))
    n_meet = sum(1 for e in valid if e.get("P_condition_number", float("inf")) <= args.p_cond_max)

    print(f"\n{'='*70}")
    print(" RECOMMENDED CONFIG  (best P conditioning)")
    print(f"{'='*70}")
    for k in ["n_rbf_centers", "rbf_width", "regularization", "q_x", "q_phi"]:
        print(f"  {k:20s} = {best[k]}")
    print(f"\n  log10(P cond)          = {log_cond:.2f}  (lower is better)")
    print(f"  P min eigenvalue       = {best['P_min_eig_real']:.3e}  (higher is better)")
    print(f"  A_cl spectral radius   = {best['A_cl_spectral_radius']:.6f}")
    print(f"  One-step RMSE (mean)   = {best['one_step_rmse_mean']:.5f}")
    print(f"  Fully controllable     = {best['controllability_full']}")
    print(f"\n  {n_meet}/{len(valid)} valid configs meet p_cond_max={args.p_cond_max:.0e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tune_results_cartpole.json"
    payload = {
        "mode": mode,
        "p_cond_max": args.p_cond_max,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "best_config": best,
        "top_valid": ranked,
        "all_valid": sorted(valid, key=_sort_key),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFull results saved to {results_path}")

    if args.retrain_best:
        retrain_and_save(best, X, X_next, U, output_dir)
    else:
        print(
            f"\nTo retrain the best config and save model + matrices, add --retrain-best\n"
            f"  python -m edmd.tune_cartpole {'--quick ' if args.quick else ''}--retrain-best"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
