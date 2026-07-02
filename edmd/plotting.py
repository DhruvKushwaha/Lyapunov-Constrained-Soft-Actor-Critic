"""
Matplotlib figures for EDMD analysis: per-component one-step prediction grids and mean multi-step rollout curves.

Used by ``edmd.train_quad_*`` and ``edmd.train_cartpole``; paths are passed in by the caller.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def save_one_step_prediction_figure(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    save_path: Path,
    state_labels: list[str],
    *,
    title: str,
    n_plot: int | None = None,
    n_cols: int = 3,
    show: bool = False,
) -> None:
    """
    Grid of true vs one-step Koopman prediction for each tracking-error component.
    """
    n_plot = min(
        n_plot if n_plot is not None else len(X_true),
        len(X_true),
        len(X_pred),
    )
    Xp = np.asarray(X_pred[:n_plot])
    Xt = np.asarray(X_true[:n_plot])
    n_comp = Xt.shape[1]
    n_rows = int(np.ceil(n_comp / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 2.8 * n_rows), sharex=True)
    axs = np.atleast_2d(axs)
    t = np.arange(n_plot)
    rmse = np.sqrt(np.mean((Xp - Xt) ** 2, axis=0))
    z0 = max(1, n_plot // 10)
    z1 = max(2, n_plot // 5)
    z2 = n_plot // 2
    z3 = min(n_plot - 1, n_plot // 2 + max(2, n_plot // 10))
    zoom_regions = [(z0, z1), (z2, z3)]

    for i in range(n_comp):
        r, c = divmod(i, n_cols)
        ax = axs[r, c]
        lbl = state_labels[i] if i < len(state_labels) else f"dim{i}"
        ax.grid(True, alpha=0.3)
        ax.plot(t, Xt[:, i], label="true", linewidth=1.5, color="blue")
        ax.plot(
            t,
            Xp[:, i],
            "--",
            label=f"pred RMSE={rmse[i]:.4f}",
            linewidth=1.5,
            color="red",
            alpha=0.85,
        )
        ax.set_ylabel(lbl, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"One-step: {lbl}", fontsize=10)
        for zs, ze in zoom_regions:
            zs = min(int(zs), n_plot - 2)
            ze = min(int(ze), n_plot - 1)
            if ze <= zs:
                continue
            ax.axvspan(zs, ze, alpha=0.08, color="gray")
            axins = inset_axes(
                ax,
                width="26%",
                height="26%",
                loc="lower left",
                bbox_to_anchor=(0.02, 0.02, 1, 1),
                bbox_transform=ax.transAxes,
            )
            m = (t >= zs) & (t <= ze)
            axins.plot(t[m], Xt[m, i], color="blue", linewidth=1.2)
            axins.plot(t[m], Xp[m, i], "--", color="red", alpha=0.85, linewidth=1.2)
            axins.set_xlim(zs, ze)
            axins.grid(True, alpha=0.3)

    for j in range(n_comp, axs.size):
        axs.flat[j].set_visible(False)

    for c in range(n_cols):
        axs[-1, c].set_xlabel("sample index", fontsize=10)
    plt.suptitle(title, y=1.01, fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved one-step prediction figure: {save_path}")


def save_multi_step_rollout_figure(
    rollout: dict,
    save_path: Path,
    *,
    title: str,
    show: bool = False,
) -> None:
    """
    Mean L2 open-loop prediction error vs horizon (multi-step with recorded controls).
    If rollout is empty or errored, still write a small diagnostic figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ys = rollout.get("mean_l2_error_by_step")
    if ys is not None and len(ys) > 0:
        h = np.arange(1, len(ys) + 1)
        y = np.array(ys, dtype=np.float64)
        ax.plot(h, y, "o-", markersize=4)
        ax.set_xlabel("Open-loop step h")
        ax.set_ylabel("Mean L2 error vs ground truth")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()
        reason = rollout.get("error", "insufficient_samples")
        n = rollout.get("n_samples", "?")
        ax.text(
            0.5,
            0.55,
            "Multi-step rollout not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
        ax.text(
            0.5,
            0.38,
            f"Reason: {reason} (n_samples={n})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title("Multi-step rollout (unavailable)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved multi-step rollout figure: {save_path}")
