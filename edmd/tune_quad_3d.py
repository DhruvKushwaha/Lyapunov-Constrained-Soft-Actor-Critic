"""
Grid-search EDMD + LQR hyperparameters for 3D quadrotor tracking-error dynamics.

Two-phase search:
  Phase 1 — EDMD phase: train one Koopman model per (n_rbf_centers, rbf_width, regularization)
             combo, compute one-step prediction RMSE.
  Phase 2 — LQR phase:  for each EDMD model, sweep (q_x, q_phi) cost weights and solve DARE.
             Only configs satisfying ALL control-theory criteria are reported as "passing".

Pass criteria (from edmd/metrics.py collect_matrix_metrics):
  - controllability_full  = True
  - A_cl_spectral_radius  < 1.0         (closed-loop stable)
  - P_min_eig_real        > 0.0         (P positive definite)
  - P_num_negative_eig    = 0
  - P_condition_number    < p_cond_max  (numerical well-conditioning)

3D quadrotor specifics:
  - State dimension:   12  (x,xd,y,yd,z,zd,phi,theta,psi,p,q,r)
  - Action dimension:  4   (motor thrusts)
  - Control frequency: 50 Hz  (dt = 1/50)
  - Lifted dim:        12 + n_rbf_centers

  NOTE: Tuning found n_rbf_centers=5 optimal (best P_cond=1.34e5 at q_x=0.01, q_phi=1e-3).
  Larger models (n_rbf>=20) over-fit and fail the P_cond threshold.

Usage
-----
  python -m edmd.tune_quad_3d --quick
  python -m edmd.tune_quad_3d
  python -m edmd.tune_quad_3d --quick --retrain-best
  python -m edmd.tune_quad_3d --data results/edmd/data/quadrotor_3d.npz --output results/edmd/quadrotor_3d
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
    "n_rbf_centers": [5, 8, 12, 16, 20],
    "rbf_width":     [0.1, 0.25, 0.5, 1.0],
    # regularization: pykoopman EDMDc uses OLS, ignores this param (empirically confirmed).
    "regularization": [1e-5],
    # q_x=1.0 is the physically correct target (matches 2D). A-normalisation (rho->1)
    # means q_x=1.0 is now feasible; old q_x=0.01 was an artefact of an unstable A.
    "q_x":           [0.1, 1.0, 10.0],
    "q_phi":         [1e-6, 1e-5, 1e-4],
}

QUICK_GRID = {
    "n_rbf_centers": [5, 8, 12],
    "rbf_width":     [0.1, 0.35, 1.0],
    "regularization": [1e-5],
    "q_x":           [1.0, 10.0],
    "q_phi":         [1e-6, 1e-5],
}

DT = 1 / 50.0
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
def _try_lqr(A, B, n_error, q_x, q_phi, p_cond_max):
    lifted_dim = A.shape[0]
    m = lifted_dim - n_error
    # Normalise A to rho <= 1 — must match the normalization in lqr.compute_lqr_gains
    # so tuner metrics reflect the same system that production training deploys.
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho_A > 1.0:
        A = A / rho_A
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

    passed = (
        metrics["controllability_full"]
        and metrics["A_cl_spectral_radius"] < 1.0
        and metrics["P_min_eig_real"] > 0.0
        and metrics["P_num_negative_eig_real"] == 0
        and p_cond < p_cond_max
    )
    return metrics, passed


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------
def run_search(X, X_next, U, grid, p_cond_max, verbose):
    n_error = X.shape[1]
    edmd_keys = ["n_rbf_centers", "rbf_width", "regularization"]
    lqr_keys  = ["q_x", "q_phi"]

    edmd_combos = list(product(*[grid[k] for k in edmd_keys]))
    lqr_combos  = list(product(*[grid[k] for k in lqr_keys]))

    n_edmd = len(edmd_combos)
    n_lqr  = len(lqr_combos)
    print(f"Search space: {n_edmd} EDMD configs × {n_lqr} LQR configs = {n_edmd * n_lqr} combos total")
    print(f"State dim: {n_error}D  (lifted dim ranges from {n_error+grid['n_rbf_centers'][0]}"
          f" to {n_error+grid['n_rbf_centers'][-1]})")

    passing = []
    failing = []
    edmd_failed = 0

    for ei, (n_rbf, width, reg) in enumerate(edmd_combos):
        tag = f"[{ei+1}/{n_edmd}] n_rbf={n_rbf}, width={width}, reg={reg:.0e}"
        if verbose:
            print(f"\n{tag} — training EDMD …", end="  ", flush=True)

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
            print(f"one-step RMSE(mean)={rmse_mean:.4f}", end="  ", flush=True)

        n_pass_lqr = 0
        for q_x, q_phi in lqr_combos:
            metrics, passed = _try_lqr(A, B, n_error, q_x, q_phi, p_cond_max)
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
                passing.append(entry)
                n_pass_lqr += 1
            else:
                failing.append(entry)

        if verbose:
            print(f"→ {n_pass_lqr}/{n_lqr} LQR configs passed")

    print(f"\nSearch complete: {len(passing)} passing / {len(passing)+len(failing)} attempted "
          f"({edmd_failed} EDMD failures skipped)")
    return passing, failing


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _sort_key(entry):
    # For LC-SAC: P_cond first (gradient quality), RMSE second, A_cl last.
    # A_cl only needs to be < 1; its exact value doesn't affect RL training.
    return (
        entry.get("P_condition_number", float("inf")),
        entry["one_step_rmse_mean"],
        entry["A_cl_spectral_radius"],
    )


def print_table(results, title, top_n=TOP_N):
    ranked = sorted(results, key=_sort_key)[:top_n]
    print(f"\n{'='*110}")
    print(f" {title}  (top {len(ranked)} shown)")
    print(f"{'='*110}")
    hdr = (
        f"{'#':>3}  {'n_rbf':>5}  {'width':>6}  {'reg':>7}  "
        f"{'q_x':>6}  {'q_phi':>8}  "
        f"{'RMSE':>7}  {'A_cl_sr':>8}  "
        f"{'P_min_eig':>10}  {'P_cond':>10}  {'ctrl':>5}"
    )
    print(hdr)
    print("-" * 110)
    for i, e in enumerate(ranked, 1):
        ctrl = "YES" if e["controllability_full"] else "NO"
        p_cond = e.get("P_condition_number", float("nan"))
        print(
            f"{i:>3}  {e['n_rbf_centers']:>5}  {e['rbf_width']:>6.2f}  "
            f"{e['regularization']:>7.0e}  {e['q_x']:>6.2f}  {e['q_phi']:>8.0e}  "
            f"{e['one_step_rmse_mean']:>7.4f}  {e['A_cl_spectral_radius']:>8.4f}  "
            f"{e['P_min_eig_real']:>10.3e}  {p_cond:>10.3e}  {ctrl:>5}"
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
    model_path    = output_dir / "edmd_model.pkl"
    matrices_path = output_dir / "lqr_matrices.npz"
    config_path   = output_dir / "best_config.json"

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
                        help="Use coarse grid (~72 combos) instead of full grid (~900)")
    parser.add_argument("--retrain-best", action="store_true",
                        help="Re-train and save artifacts for the best passing config")
    parser.add_argument("--p-cond-max", type=float, default=1e7,
                        help="Max allowed P condition number (default: 1e7)")
    parser.add_argument("--data", type=str, default="results/edmd/data/quadrotor_3d.npz",
                        help="Path to .npz data file")
    parser.add_argument("--output", type=str, default="results/edmd/quadrotor_3d",
                        help="Output directory for results JSON and best-config artifacts")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-EDMD-config progress")
    args = parser.parse_args()

    data_file = Path(args.data)
    output_dir = Path(args.output)

    if not data_file.is_file():
        print(
            f"Data file not found: {data_file}\n"
            "Run: python -m edmd.collect_quad_3d\nthen retry.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading data from {data_file} …")
    raw = np.load(data_file)
    X      = np.asarray(raw["tracking_error"],      dtype=np.float64)
    X_next = np.asarray(raw["tracking_error_next"], dtype=np.float64)
    U      = np.asarray(raw["U"],                   dtype=np.float64)
    print(f"  X={X.shape}, X_next={X_next.shape}, U={U.shape}")

    grid = QUICK_GRID if args.quick else FULL_GRID
    mode = "QUICK" if args.quick else "FULL"
    print(f"\nUsing {mode} grid.  P condition-number limit: {args.p_cond_max:.0e}")

    passing, failing = run_search(X, X_next, U, grid, args.p_cond_max, args.verbose)

    if not passing:
        print(
            "\nNo passing configs found.\n"
            "Suggestions:\n"
            "  • Relax --p-cond-max\n"
            "  • Try larger n_rbf_centers (3D quad needs more lifting for controllability)\n"
            "  • Try smaller rbf_width\n"
            "  • Check data quality: run edmd.collect_quad_3d with more episodes/trajectories"
        )
        print_table(failing, "TOP FAILING CONFIGS (closest to passing)", top_n=TOP_N)
        return 1

    ranked = print_table(passing, "TOP PASSING CONFIGS (all control-theory criteria met)", top_n=TOP_N)
    best = ranked[0]

    print(f"\n{'='*60}")
    print(" RECOMMENDED CONFIG")
    print(f"{'='*60}")
    for k in ["n_rbf_centers", "rbf_width", "regularization", "q_x", "q_phi"]:
        print(f"  {k:20s} = {best[k]}")
    print(f"\n  One-step RMSE (mean)   = {best['one_step_rmse_mean']:.5f}")
    print(f"  A_cl spectral radius   = {best['A_cl_spectral_radius']:.6f}")
    print(f"  P min eigenvalue       = {best['P_min_eig_real']:.3e}")
    print(f"  P condition number     = {best.get('P_condition_number', float('nan')):.3e}")
    print(f"  Fully controllable     = {best['controllability_full']}")
    print(f"  Lifted dimension       = {best['lifted_dim']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tune_results.json"
    payload = {
        "mode": mode,
        "p_cond_max": args.p_cond_max,
        "n_passing": len(passing),
        "n_failing": len(failing),
        "best_config": best,
        "top_passing": ranked,
        "all_passing": sorted(passing, key=_sort_key),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFull results saved to {results_path}")

    if args.retrain_best:
        retrain_and_save(best, X, X_next, U, output_dir)

    if not args.retrain_best:
        print(
            f"\nTo retrain the best config and save model + matrices, add --retrain-best\n"
            f"  python -m edmd.tune_quad_3d {'--quick ' if args.quick else ''}--retrain-best"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
