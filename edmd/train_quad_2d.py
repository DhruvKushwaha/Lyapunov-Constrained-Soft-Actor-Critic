"""
Fit EDMDc on 2D quadrotor **tracking error** and export LQR matrices + diagnostics.

Requires ``Saved_data/data_EDMD_2D.npz`` (run ``python -m edmd.collect_quad_2d`` first); exits with code 2
if the file is missing. Loads optional ``episode_id`` for rollout metrics.

Set ``EDMD_SHOW_PLOTS=0`` to run headless (figures are still written to ``Saved_data/``).

CLI flags (all optional — defaults match the original hard-coded values):
  --n-rbf-centers INT    Number of RBF centres (default: 2, giving N_lift=8)
  --rbf-width FLOAT      RBF kernel width (default: 0.25)
  --regularization FLOAT EDMDc regularisation strength (default: 1e-5)
  --q-x FLOAT            LQR cost weight on physical error states (default: 1.0)
  --q-phi FLOAT          LQR cost weight on lifted-only coordinates (default: 1e-6)
  --data PATH            Path to .npz data file (default: Saved_data/data_EDMD_2D.npz)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import matplotlib.pyplot as plt
import numpy as np
import pykoopman as pk
from pykoopman.regression import EDMDc
from sklearn.cluster import KMeans

from edmd.lqr import analyze_p_matrix, compute_lqr_gains
from edmd.metrics import collect_matrix_metrics, one_step_metrics, rollout_errors
from edmd.plotting import save_multi_step_rollout_figure, save_one_step_prediction_figure

STATE_LABELS_2D = ["x", "x_dot", "z", "z_dot", "theta", "theta_dot"]


def load_data(data_file):
    print("--- Loading Data ---")
    data = np.load(data_file)

    tracking_error = np.asarray(data["tracking_error"], dtype=np.float64)
    tracking_error_next = np.asarray(data["tracking_error_next"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)

    print(f"Loaded tracking_error shape: {tracking_error.shape}")
    print(f"Loaded tracking_error_next shape: {tracking_error_next.shape}")
    print(f"Loaded U shape: {U.shape}")

    out = {
        "tracking_error": tracking_error,
        "tracking_error_next": tracking_error_next,
        "U": U,
    }
    if "episode_id" in data.files:
        out["episode_id"] = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
        print(f"Loaded episode_id shape: {out['episode_id'].shape}")
    return out


def train_edmd_model(X, X_next, U, n_rbf_centers=2, rbf_width=0.25, regularization=1e-5, dt=1 / 50.0):
    print("--- Training Koopman Operator on Tracking Error ---")

    regressor = EDMDc()

    kmeans = KMeans(n_clusters=n_rbf_centers, random_state=42, n_init=12)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers_pk = centers.T

    RBF = pk.observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=centers_pk.shape[1],
        centers=centers_pk,
        kernel_width=rbf_width,
        polyharmonic_coeff=1.0,
        include_state=True,
    )

    print(
        f"Tracking error (X): {X.shape}, Tracking error next (X_next): {X_next.shape}, "
        f"U: {U.shape}, centers: {centers_pk.shape}"
    )

    model = pk.Koopman(observables=RBF, regressor=regressor, quiet=True)
    model.fit(X, y=X_next, u=U, dt=dt)

    return model, RBF


def plot_p_matrix_analysis(P_lifted, save_dir, show=True, prefix="edmd_2d_"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    P_to_plot = P_lifted
    P_plot_name = "P_lifted"

    ax1 = axes[0]
    im1 = ax1.imshow(P_to_plot, cmap="viridis", aspect="auto")
    ax1.set_title(f"{P_plot_name} Matrix Heatmap")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Value")

    ax2 = axes[1]
    eigenvals = np.linalg.eigvals(P_to_plot)
    eigenvals_real = np.real(eigenvals)
    eigenvals_imag = np.imag(eigenvals)
    ax2.scatter(eigenvals_real, eigenvals_imag, alpha=0.6, s=20)
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_title(f"{P_plot_name} Eigenvalues")
    ax2.set_xlabel("Real Part")
    ax2.set_ylabel("Imaginary Part")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    eigenvals_sorted = np.sort(eigenvals_real)[::-1]
    ax3.plot(eigenvals_sorted, "o-", markersize=3)
    ax3.set_title(f"{P_plot_name} Eigenvalues (Sorted)")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Eigenvalue (Real Part)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()

    plot_path = save_dir / f"{prefix}P_matrix_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved P_matrix analysis plot to {plot_path}")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_model(model, X, X_next, U, n_plot=100):
    X_s = X[:n_plot]
    U_s = U[:n_plot]
    X_next_s = X_next[:n_plot]

    X_pred_koop = model.predict(X_s, u=U_s)
    X_pred_koop = np.asarray(X_pred_koop)

    assert X_pred_koop.shape == X_next_s.shape

    rmse_koop = np.sqrt(np.mean((X_pred_koop - X_next_s) ** 2, axis=0))

    print("RMSE (6 tracking error components for 2D quadrotor) - Koopman:", np.round(rmse_koop, 4))

    return rmse_koop, X_pred_koop, X_next_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rbf-centers", type=int,   default=2,    help="Number of RBF centres (default: 2, giving N_lift=8 = 2 RBF + 6 state)")
    parser.add_argument("--rbf-width",     type=float, default=0.25, help="RBF kernel width (default: 0.25)")
    parser.add_argument("--regularization",type=float, default=1e-5, help="EDMDc regularisation (default: 1e-5)")
    parser.add_argument("--q-x",           type=float, default=1.0,  help="LQR weight on physical error states (default: 1.0)")
    parser.add_argument("--q-phi",         type=float, default=1e-6, help="LQR weight on lifted-only coords (default: 1e-6)")
    parser.add_argument("--data",          type=str,   default=None, help="Path to .npz data file (default: results/edmd/data/quadrotor_2d_track.npz)")
    parser.add_argument("--model-output",  type=str,   default=None, help="Path to save Koopman model .pkl (default: results/edmd/quadrotor_2d_track/edmd_model.pkl)")
    parser.add_argument("--riccati-output",type=str,   default=None, help="Path to save LQR matrices .npz (default: results/edmd/quadrotor_2d_track/lqr_matrices.npz)")
    parser.add_argument("--metrics-prefix",type=str,   default="edmd_2d_", help="Prefix for diagnostic output filenames (default: 'edmd_2d_'). Set to 'edmd_2d_stab_' for stabilization runs.")
    args = parser.parse_args()

    show_plots = os.environ.get("EDMD_SHOW_PLOTS", "1").lower() in ("1", "true", "yes")
    SAVE_DATA_DIR = REPO_ROOT / "results" / "edmd" / "quadrotor_2d_track"
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Save data directory: {SAVE_DATA_DIR.absolute()}")

    DATA_FILE = Path(args.data) if args.data else REPO_ROOT / "results" / "edmd" / "data" / "quadrotor_2d_track.npz"
    MODEL_SAVE_PATH = Path(args.model_output) if args.model_output else SAVE_DATA_DIR / "edmd_model.pkl"
    RICCATI_SAVE_PATH = Path(args.riccati_output) if args.riccati_output else SAVE_DATA_DIR / "lqr_matrices.npz"

    if not DATA_FILE.is_file():
        print(
            f"Data file not found: {DATA_FILE}\n"
            "Run: python -m edmd.collect_quad_2d\n"
            "then retry.",
            file=sys.stderr,
        )
        return 2

    N_RBF_CENTERS = args.n_rbf_centers
    RBF_WIDTH = args.rbf_width
    REGULARIZATION = args.regularization
    PREFIX = args.metrics_prefix
    DT = 1 / 50.0
    ROLLOUT_HORIZON = 15
    N_ROLLOUT_STARTS = 40
    ROLLOUT_SEED = 0

    print(f"EDMD config: n_rbf_centers={N_RBF_CENTERS}, rbf_width={RBF_WIDTH}, "
          f"regularization={REGULARIZATION:.0e}, q_x={args.q_x}, q_phi={args.q_phi:.0e}")

    data = load_data(DATA_FILE)
    X = data["tracking_error"]
    X_next = data["tracking_error_next"]
    U = data["U"]
    episode_id = data.get("episode_id")

    model, RBF = train_edmd_model(
        X,
        X_next,
        U,
        n_rbf_centers=N_RBF_CENTERS,
        rbf_width=RBF_WIDTH,
        regularization=REGULARIZATION,
        dt=DT,
    )

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved Koopman model to {MODEL_SAVE_PATH}")

    A_lifted = model.A
    B_lifted = model.B
    lifted_dim = A_lifted.shape[0]

    print(f"Extracted A_lifted shape: {A_lifted.shape}")
    print(f"Extracted B_lifted shape: {B_lifted.shape}")
    print(f"Lifted space dimension: {lifted_dim}")

    n_error = X.shape[1]
    lqr_results = compute_lqr_gains(A_lifted, B_lifted, n_error=n_error, q_x=args.q_x, q_phi=args.q_phi)

    P_lifted = lqr_results["P"]
    K = lqr_results["K"]
    Q = lqr_results["Q"]
    R = lqr_results["R"]
    A_cl = lqr_results["A_cl"]
    A_lifted = lqr_results["A_normalized"]  # use normalised A that P was designed against

    print("\n--- Saving fixed matrices ---")
    save_dict = {
        "A_lifted": A_lifted,
        "B_lifted": B_lifted,
        "A_cl": A_cl,
        "P": P_lifted,
        "K": K,
        "Q": Q,
        "R": R,
    }

    np.savez(RICCATI_SAVE_PATH, **save_dict)
    print(f"✓ Saved fixed Riccati/LQR matrices to {RICCATI_SAVE_PATH}")
    print(f"  A_lifted ({A_lifted.shape}), B_lifted ({B_lifted.shape}), P_lifted ({P_lifted.shape})")
    print(f"\nNote: Matrices represent tracking error dynamics (not state dynamics)")
    print(f"  - Normalized Q and R matrices")
    print(f"  - LQR gain K minimizes tracking error")

    analyze_p_matrix(P_lifted, A_lifted, B_lifted)

    p_fig = SAVE_DATA_DIR / f"{PREFIX}P_matrix_analysis.png"
    plot_p_matrix_analysis(P_lifted, SAVE_DATA_DIR, show=show_plots, prefix=PREFIX)

    print("\n--- Evaluating Model Predictions ---")
    n_plot_vis = 100
    rmse, X_pred, X_true = evaluate_model(model, X, X_next, U, n_plot=n_plot_vis)
    print(f"RMSE: {rmse}")

    one_step_fig = SAVE_DATA_DIR / f"{PREFIX}one_step_prediction.png"
    save_one_step_prediction_figure(
        X_pred,
        X_true,
        one_step_fig,
        STATE_LABELS_2D[:n_error],
        title="EDMD 2D: one-step prediction vs ground truth (all components)",
        n_plot=n_plot_vis,
        show=show_plots,
    )

    matrix_metrics = collect_matrix_metrics(A_lifted, B_lifted, P_lifted, K, A_cl, n_error)
    one_step_full = one_step_metrics(
        model, X, X_next, U, state_labels=STATE_LABELS_2D[: X.shape[1]]
    )
    rollout = rollout_errors(
        model,
        X,
        X_next,
        U,
        horizon=ROLLOUT_HORIZON,
        n_starts=N_ROLLOUT_STARTS,
        seed=ROLLOUT_SEED,
        episode_id=episode_id,
    )
    roll_fig = SAVE_DATA_DIR / f"{PREFIX}multi_step_rollout.png"
    save_multi_step_rollout_figure(
        rollout,
        roll_fig,
        title="EDMD 2D: multi-step rollout (open loop, recorded u)",
        show=show_plots,
    )

    metrics_path = SAVE_DATA_DIR / f"{PREFIX}metrics.json"
    metrics_payload = {
        "data_file": str(DATA_FILE.resolve()),
        "n_samples": int(len(X)),
        "dt": DT,
        "n_rbf_centers": N_RBF_CENTERS,
        "rbf_width": RBF_WIDTH,
        "regularization": REGULARIZATION,
        "q_x": args.q_x,
        "q_phi": args.q_phi,
        "rollout_horizon": ROLLOUT_HORIZON,
        "n_rollout_starts": N_ROLLOUT_STARTS,
        "rollout_seed": ROLLOUT_SEED,
        "matrix_metrics": matrix_metrics,
        "one_step_full_dataset": one_step_full,
        "rollout_open_loop": rollout,
        "artifacts": {
            "model_pkl": str(MODEL_SAVE_PATH.resolve()),
            "matrices_npz": str(RICCATI_SAVE_PATH.resolve()),
            "figures": {
                "P_matrix": str(p_fig.resolve()),
                "one_step_prediction": str(one_step_fig.resolve()),
                "multi_step_rollout": str(roll_fig.resolve()),
                "rollout": str(roll_fig.resolve()),
            },
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    print("\nEDMD training completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
