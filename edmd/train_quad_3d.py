"""
Fit EDMDc on 3D quadrotor tracking error; same artifact layout as ``train_quad_2d``.

Expects ``results/edmd/data/quadrotor_3d.npz`` (``python -m edmd.collect_quad_3d``). Missing file → exit code 2.
Optional ``episode_id`` in the archive enables episode-aware rollout metrics.

CLI flags (all optional — defaults are the tuning-optimal values from edmd.tune_quad_3d):
  --n-rbf-centers INT    Number of RBF centres (default: 5)
  --rbf-width FLOAT      RBF kernel width (default: 0.25)
  --regularization FLOAT EDMDc regularisation strength (default: 1e-5)
  --q-x FLOAT            LQR cost weight on physical error states (default: 0.01)
  --q-phi FLOAT          LQR cost weight on lifted-only coordinates (default: 1e-3)
  --data PATH            Path to .npz data file (default: results/edmd/data/quadrotor_3d.npz)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pykoopman as pk
from pykoopman.observables import RadialBasisFunction
from pykoopman.regression import EDMDc
from sklearn.cluster import KMeans

from edmd.lqr import analyze_p_matrix, compute_lqr_gains
from edmd.metrics import collect_matrix_metrics, one_step_metrics, rollout_errors
from edmd.plotting import save_multi_step_rollout_figure, save_one_step_prediction_figure

STATE_LABELS_3D = [
    "x",
    "x_dot",
    "y",
    "y_dot",
    "z",
    "z_dot",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
]

EXPECTED_N_ERROR = 12
EXPECTED_N_U = 4


def load_data(data_file: Path) -> dict:
    print("--- Loading Data (3D) ---")
    data = np.load(data_file)
    tracking_error = np.asarray(data["tracking_error"], dtype=np.float64)
    tracking_error_next = np.asarray(data["tracking_error_next"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    print(f"Loaded tracking_error shape: {tracking_error.shape}")
    print(f"Loaded tracking_error_next shape: {tracking_error_next.shape}")
    print(f"Loaded U shape: {U.shape}")

    ne = tracking_error.shape[1]
    if ne != EXPECTED_N_ERROR:
        print(
            f"Warning: expected {EXPECTED_N_ERROR}D tracking error (3D quad), got {ne}.",
            file=sys.stderr,
        )
    if U.shape[1] != EXPECTED_N_U:
        print(
            f"Warning: expected U with {EXPECTED_N_U} columns (3D quad motors), got {U.shape[1]}.",
            file=sys.stderr,
        )

    out = {
        "tracking_error": tracking_error,
        "tracking_error_next": tracking_error_next,
        "U": U,
    }
    if "episode_id" in data.files:
        out["episode_id"] = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
        print(f"Loaded episode_id shape: {out['episode_id'].shape}")
    return out


def train_edmd_model(X, X_next, U, n_rbf_centers=8, rbf_width=0.35, regularization=1e-5, dt=1 / 50.0):
    print("--- Training Koopman Operator on 3D Tracking Error ---")
    regressor = EDMDc()
    kmeans = KMeans(n_clusters=n_rbf_centers, random_state=42, n_init=12)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers_pk = centers.T

    RBF = RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=centers_pk.shape[1],
        centers=centers_pk,
        kernel_width=rbf_width,
        polyharmonic_coeff=1.0,
        include_state=True,
    )

    print(
        f"Tracking error (X): {X.shape}, X_next: {X_next.shape}, "
        f"U: {U.shape}, centers: {centers_pk.shape}"
    )

    model = pk.Koopman(observables=RBF, regressor=regressor, quiet=True)
    model.fit(X, y=X_next, u=U, dt=dt)
    return model, RBF


def plot_p_matrix_analysis(P_lifted, save_dir: Path, show: bool = True) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    P_to_plot = P_lifted
    P_plot_name = "P_lifted"

    ax1 = axes[0]
    im1 = ax1.imshow(P_to_plot, cmap="viridis", aspect="auto")
    ax1.set_title(f"{P_plot_name} Matrix Heatmap (3D quad EDMD)")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Value")

    ax2 = axes[1]
    eigenvals = np.linalg.eigvals(P_to_plot)
    ax2.scatter(np.real(eigenvals), np.imag(eigenvals), alpha=0.6, s=20)
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_title(f"{P_plot_name} Eigenvalues")
    ax2.set_xlabel("Real Part")
    ax2.set_ylabel("Imaginary Part")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    eigenvals_sorted = np.sort(np.real(eigenvals))[::-1]
    ax3.plot(eigenvals_sorted, "o-", markersize=3)
    ax3.set_title(f"{P_plot_name} Eigenvalues (Sorted, Real)")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Eigenvalue (Real Part)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plot_path = save_dir / "EDMD_3D_P_matrix_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved P_matrix analysis plot to {plot_path}")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_model(model, X, X_next, U, n_plot=100):
    X_s = X[:n_plot]
    U_s = U[:n_plot]
    X_next_s = X_next[:n_plot]
    X_pred_koop = np.asarray(model.predict(X_s, u=U_s))
    assert X_pred_koop.shape == X_next_s.shape
    rmse_koop = np.sqrt(np.mean((X_pred_koop - X_next_s) ** 2, axis=0))
    print(
        f"RMSE ({X.shape[1]} tracking error components, 3D quad) - Koopman:",
        np.round(rmse_koop, 4),
    )
    return rmse_koop, X_pred_koop, X_next_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rbf-centers", type=int,   default=5,    help="Number of RBF centres (default: 5)")
    parser.add_argument("--rbf-width",     type=float, default=0.25, help="RBF kernel width (default: 0.25)")
    parser.add_argument("--regularization",type=float, default=1e-5, help="EDMDc regularisation (default: 1e-5)")
    parser.add_argument("--q-x",           type=float, default=1.0,  help="LQR weight on physical error states (default: 1.0)")
    parser.add_argument("--q-phi",         type=float, default=1e-6, help="LQR weight on lifted-only coords (default: 1e-6)")
    parser.add_argument("--data",          type=str,   default=None, help="Path to .npz data file (default: results/edmd/data/quadrotor_3d.npz)")
    parser.add_argument("--output",        type=str,   default=None, help="Output directory for model/matrix artifacts (default: results/edmd/quadrotor_3d)")
    args = parser.parse_args()

    show_plots = os.environ.get("EDMD_SHOW_PLOTS", "1").lower() in ("1", "true", "yes")
    repo_root = Path(__file__).resolve().parent.parent
    save_dir = Path(args.output) if args.output else repo_root / "results" / "edmd" / "quadrotor_3d"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save data directory: {save_dir.absolute()}")

    data_file = Path(args.data) if args.data else repo_root / "results" / "edmd" / "data" / "quadrotor_3d.npz"
    model_path = save_dir / "edmd_model.pkl"
    riccati_path = save_dir / "lqr_matrices.npz"

    n_rbf_centers = args.n_rbf_centers
    rbf_width = args.rbf_width
    dt = 1 / 50.0
    rollout_horizon = 15
    n_rollout_starts = 40
    rollout_seed = 0
    n_plot_vis = 100

    if not data_file.is_file():
        print(
            f"Data file not found: {data_file}\n"
            "Run: python -m edmd.collect_quad_3d\n"
            "then retry.",
            file=sys.stderr,
        )
        return 2

    print(f"EDMD config: n_rbf_centers={n_rbf_centers}, rbf_width={rbf_width}, "
          f"regularization={args.regularization:.0e}, q_x={args.q_x}, q_phi={args.q_phi:.0e}")

    data = load_data(data_file)
    X = data["tracking_error"]
    X_next = data["tracking_error_next"]
    U = data["U"]
    episode_id = data.get("episode_id")
    n_error = X.shape[1]

    model, _ = train_edmd_model(
        X, X_next, U, n_rbf_centers=n_rbf_centers, rbf_width=rbf_width,
        regularization=args.regularization, dt=dt,
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved Koopman model to {model_path}")

    A_lifted = model.A
    B_lifted = model.B
    print(f"Extracted A_lifted shape: {A_lifted.shape}")
    print(f"Extracted B_lifted shape: {B_lifted.shape}")

    lqr_results = compute_lqr_gains(A_lifted, B_lifted, n_error=n_error, q_x=args.q_x, q_phi=args.q_phi)
    P_lifted = lqr_results["P"]
    K = lqr_results["K"]
    Q = lqr_results["Q"]
    R_mat = lqr_results["R"]
    A_cl = lqr_results["A_cl"]
    # Use the (possibly normalised) A that P was actually designed against.
    # RL agents load A_lifted from this file for forward prediction z' = A z + B u.
    A_lifted = lqr_results["A_normalized"]

    np.savez(
        riccati_path,
        A_lifted=A_lifted,
        B_lifted=B_lifted,
        A_cl=A_cl,
        P=P_lifted,
        K=K,
        Q=Q,
        R=R_mat,
    )
    print(f"Saved Riccati/LQR matrices to {riccati_path}")

    analyze_p_matrix(P_lifted, A_lifted, B_lifted)
    plot_p_matrix_analysis(P_lifted, save_dir, show=show_plots)

    print("\n--- Evaluating Model Predictions ---")
    rmse, X_pred, X_true = evaluate_model(model, X, X_next, U, n_plot=n_plot_vis)
    print(f"RMSE (subset): {rmse}")
    labels = STATE_LABELS_3D[:n_error]
    pred_fig = save_dir / "edmd_3d_one_step_prediction.png"
    save_one_step_prediction_figure(
        X_pred,
        X_true,
        pred_fig,
        labels,
        title="EDMD 3D: one-step prediction vs ground truth (all components)",
        n_plot=n_plot_vis,
        show=show_plots,
    )

    matrix_metrics = collect_matrix_metrics(A_lifted, B_lifted, P_lifted, K, A_cl, n_error)
    one_step_full = one_step_metrics(model, X, X_next, U, state_labels=labels)
    rollout = rollout_errors(
        model,
        X,
        X_next,
        U,
        horizon=rollout_horizon,
        n_starts=n_rollout_starts,
        seed=rollout_seed,
        episode_id=episode_id,
    )
    roll_fig = save_dir / "edmd_3d_multi_step_rollout.png"
    save_multi_step_rollout_figure(
        rollout,
        roll_fig,
        title="EDMD 3D: multi-step rollout (open loop, recorded u)",
        show=show_plots,
    )

    p_fig = save_dir / "EDMD_3D_P_matrix_analysis.png"
    metrics_path = save_dir / "edmd_3d_metrics.json"
    metrics_payload = {
        "data_file": str(data_file.resolve()),
        "n_samples": int(len(X)),
        "dt": dt,
        "n_rbf_centers": n_rbf_centers,
        "rbf_width": rbf_width,
        "regularization": args.regularization,
        "q_x": args.q_x,
        "q_phi": args.q_phi,
        "rollout_horizon": rollout_horizon,
        "n_rollout_starts": n_rollout_starts,
        "rollout_seed": rollout_seed,
        "matrix_metrics": matrix_metrics,
        "one_step_full_dataset": one_step_full,
        "rollout_open_loop": rollout,
        "artifacts": {
            "model_pkl": str(model_path.resolve()),
            "matrices_npz": str(riccati_path.resolve()),
            "figures": {
                "P_matrix": str(p_fig.resolve()),
                "one_step_prediction": str(pred_fig.resolve()),
                "multi_step_rollout": str(roll_fig.resolve()),
                "rollout": str(roll_fig.resolve()),
            },
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print("\nEDMD 3D training completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
