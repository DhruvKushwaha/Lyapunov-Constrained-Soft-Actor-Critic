"""
Fit EDMDc on Cartpole tracking error; writes model, LQR ``.npz``, figures, and ``edmd_cartpole_metrics.json``.

Dataset from ``python -m edmd.collect_cartpole``. Supports optional ``episode_id`` for rollout metrics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pykoopman as pk
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pykoopman.regression import EDMDc
from sklearn.cluster import KMeans

from edmd.lqr import analyze_p_matrix, compute_lqr_gains
from edmd.metrics import collect_matrix_metrics, one_step_metrics, rollout_errors
from edmd.plotting import save_multi_step_rollout_figure

REPO_ROOT = Path(__file__).resolve().parent.parent

STATE_LABELS = ["x", "x_dot", "theta", "theta_dot"]


def load_data(path: Path) -> dict:
    print("--- Loading Data ---")
    data = np.load(path)
    te = np.asarray(data["tracking_error"], dtype=np.float64)
    tn = np.asarray(data["tracking_error_next"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    print(f"tracking_error: {te.shape}, tracking_error_next: {tn.shape}, U: {U.shape}")
    if te.shape[1] != 4:
        print(f"Warning: expected 4D cartpole error, got {te.shape[1]}", file=sys.stderr)
    out = {"tracking_error": te, "tracking_error_next": tn, "U": U}
    if "episode_id" in data.files:
        out["episode_id"] = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
        print(f"Loaded episode_id shape: {out['episode_id'].shape}")
    return out


def train_edmd_model(X, X_next, U, n_rbf_centers, rbf_width, dt, regularization=1e-5):
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
    print(f"X: {X.shape}, X_next: {X_next.shape}, U: {U.shape}, centers: {centers_pk.shape}")
    model = pk.Koopman(observables=RBF, regressor=regressor, quiet=True)
    model.fit(X, y=X_next, u=U, dt=dt)
    return model, RBF


def save_p_matrix_figure(P: np.ndarray, save_path: Path, *, show: bool) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    im1 = axes[0].imshow(P, cmap="viridis", aspect="auto")
    axes[0].set_title("P_lifted heatmap")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[0], label="Value")
    ev = np.linalg.eigvals(P)
    axes[1].scatter(np.real(ev), np.imag(ev), alpha=0.6, s=20)
    axes[1].axhline(0, color="r", linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="r", linestyle="--", alpha=0.5)
    axes[1].set_title("P_lifted eigenvalues")
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Imag")
    axes[1].grid(True, alpha=0.3)
    evr = np.sort(np.real(ev))[::-1]
    axes[2].plot(evr, "o-", markersize=3)
    axes[2].set_title("P_lifted eigenvalues (real, sorted)")
    axes[2].axhline(0, color="r", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {save_path}")


def save_prediction_figure(X_pred, X_true, save_path: Path, *, n_plot: int, show: bool) -> None:
    n_plot = min(n_plot, len(X_true), len(X_pred))
    Xp = X_pred[:n_plot]
    Xt = X_true[:n_plot]
    n_comp = Xt.shape[1]
    n_rows = int(np.ceil(n_comp / 2))
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), sharex=True)
    axs = np.atleast_2d(axs)
    t = np.arange(n_plot)
    rmse = np.sqrt(np.mean((Xp - Xt) ** 2, axis=0))
    zoom_regions = [
        (max(1, n_plot // 10), max(2, n_plot // 5)),
        (n_plot // 2, min(n_plot - 1, n_plot // 2 + n_plot // 10)),
    ]

    for i in range(n_comp):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        lbl = STATE_LABELS[i] if i < len(STATE_LABELS) else f"dim{i}"
        ax.grid(True, alpha=0.3)
        ax.plot(t, Xt[:, i], label="true", linewidth=2, color="blue")
        ax.plot(t, Xp[:, i], "--", label=f"pred RMSE={rmse[i]:.4f}", linewidth=2, color="red", alpha=0.85)
        ax.set_ylabel(lbl)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"One-step: {lbl}")
        for zs, ze in zoom_regions:
            zs = min(zs, n_plot - 2)
            ze = min(ze, n_plot - 1)
            if ze <= zs:
                continue
            ax.axvspan(zs, ze, alpha=0.08, color="gray")
            axins = inset_axes(
                ax,
                width="28%",
                height="28%",
                loc="lower left",
                bbox_to_anchor=(0.02, 0.02, 1, 1),
                bbox_transform=ax.transAxes,
            )
            m = (t >= zs) & (t <= ze)
            axins.plot(t[m], Xt[m, i], color="blue", linewidth=1.5)
            axins.plot(t[m], Xp[m, i], "--", color="red", alpha=0.85, linewidth=1.5)
            axins.set_xlim(zs, ze)
            axins.grid(True, alpha=0.3)
    for j in range(n_comp, axs.size):
        axs.flat[j].set_visible(False)
    axs[-1, 0].set_xlabel("sample index")
    if n_comp > 1:
        axs[-1, min(1, n_comp - 1)].set_xlabel("sample index")
    plt.suptitle("EDMD cartpole: one-step prediction vs ground truth (all components)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", type=str, default="results/edmd/data/cartpole.npz")
    p.add_argument("--output-dir", type=str, default="results/edmd/cartpole")
    p.add_argument("--dt", type=float, default=1.0 / 15.0, help="Control timestep (default 1/15 for cartpole yaml).")
    p.add_argument("--n-rbf-centers", type=int, default=5)
    p.add_argument("--rbf-width", type=float, default=0.35)
    p.add_argument("--regularization", type=float, default=1e-5, help="EDMDc regularisation strength (default: 1e-5)")
    p.add_argument("--n-plot", type=int, default=150, help="Samples in prediction comparison figure.")
    p.add_argument("--rollout-horizon", type=int, default=15)
    p.add_argument("--n-rollout-starts", type=int, default=40)
    p.add_argument("--rollout-seed", type=int, default=0)
    p.add_argument("--q-x", type=float, default=1.0, help="LQR Q weight on physical error coords.")
    p.add_argument("--q-phi", type=float, default=1e-6, help="LQR Q weight on lifted coords.")
    p.add_argument("--show-plots", action="store_true", help="Also call plt.show() after each figure.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_file)
    if not data_path.is_file():
        rep = REPO_ROOT / args.data_file
        data_path = rep if rep.is_file() else data_path
    if not data_path.is_file():
        print(
            f"Data file not found: {args.data_file}\n"
            "Run: python -m edmd.collect_cartpole\n"
            "then retry.",
            file=sys.stderr,
        )
        return 2

    data = load_data(data_path)
    X = data["tracking_error"]
    X_next = data["tracking_error_next"]
    U = data["U"]
    episode_id = data.get("episode_id")
    n_error = X.shape[1]

    model, _ = train_edmd_model(
        X,
        X_next,
        U,
        n_rbf_centers=args.n_rbf_centers,
        rbf_width=args.rbf_width,
        dt=args.dt,
        regularization=args.regularization,
    )

    model_path = out_dir / "edmd_model_cartpole.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved Koopman model to {model_path}")

    A = model.A
    B = model.B
    lqr = compute_lqr_gains(A, B, n_error=n_error, q_x=args.q_x, q_phi=args.q_phi)
    P, K, Q, R_mat, A_cl = lqr["P"], lqr["K"], lqr["Q"], lqr["R"], lqr["A_cl"]
    A = lqr["A_normalized"]  # use normalised A that P was designed against

    riccati_path = out_dir / "lqr_matrices_cartpole.npz"
    np.savez(
        riccati_path,
        A_lifted=A,
        B_lifted=B,
        A_cl=A_cl,
        P=P,
        K=K,
        Q=Q,
        R=R_mat,
    )
    print(f"Saved matrices to {riccati_path}")

    matrix_metrics = collect_matrix_metrics(A, B, P, K, A_cl, n_error)
    one_step = one_step_metrics(model, X, X_next, U, state_labels=STATE_LABELS)
    rollout = rollout_errors(
        model,
        X,
        X_next,
        U,
        horizon=args.rollout_horizon,
        n_starts=args.n_rollout_starts,
        seed=args.rollout_seed,
        episode_id=episode_id,
    )

    analyze_p_matrix(P, A, B)

    p_fig = out_dir / "edmd_cartpole_P_matrix_analysis.png"
    save_p_matrix_figure(P, p_fig, show=args.show_plots)

    n_plot = min(args.n_plot, len(X))
    X_pred_vis = np.asarray(model.predict(X[:n_plot], u=U[:n_plot]))
    pred_fig = out_dir / "edmd_cartpole_one_step_prediction.png"
    save_prediction_figure(
        X_pred_vis, X_next[:n_plot], pred_fig, n_plot=n_plot, show=args.show_plots
    )

    roll_fig = out_dir / "edmd_cartpole_multi_step_rollout.png"
    save_multi_step_rollout_figure(
        rollout,
        roll_fig,
        title="EDMD cartpole: multi-step rollout (open loop, recorded u)",
        show=args.show_plots,
    )

    metrics_path = out_dir / "edmd_cartpole_metrics.json"
    payload = {
        "data_file": str(data_path.resolve()),
        "n_samples": int(len(X)),
        "dt": args.dt,
        "n_rbf_centers": args.n_rbf_centers,
        "rbf_width": args.rbf_width,
        "regularization": args.regularization,
        "rollout_horizon": args.rollout_horizon,
        "n_rollout_starts": args.n_rollout_starts,
        "rollout_seed": args.rollout_seed,
        "q_x": args.q_x,
        "q_phi": args.q_phi,
        "matrix_metrics": matrix_metrics,
        "one_step_full_dataset": one_step,
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
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print("EDMD cartpole training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
