"""
Metrics for fitted EDMD / Koopman models (JSON-serializable).

Includes lifted-space matrix summaries, full-dataset one-step error, and **open-loop rollout** error vs
recorded controls. Pass ``episode_id`` into ``rollout_errors`` when the dataset labels episodes so
rollout starts never cross episode boundaries; omit for legacy behavior.
"""

from __future__ import annotations

import numpy as np


def _discrete_controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Controllability matrix [B, AB, ..., A^{n-1}B] (same as MATLAB/Python-control ctrb)."""
    n = A.shape[0]
    cols = [B]
    ak_b = B
    for _ in range(1, n):
        ak_b = A @ ak_b
        cols.append(ak_b)
    return np.hstack(cols)


def collect_matrix_metrics(A, B, P, K, A_cl, n_error: int) -> dict:
    lifted_dim = A.shape[0]
    eig_a = np.linalg.eigvals(A)
    eig_acl = np.linalg.eigvals(A_cl)
    eig_p = np.linalg.eigvals(P)
    ctrb = _discrete_controllability_matrix(A, B)
    rank = int(np.linalg.matrix_rank(ctrb))
    sym_p = bool(np.allclose(P, P.T, rtol=1e-5, atol=1e-8))
    return {
        "lifted_dim": int(lifted_dim),
        "n_error": int(n_error),
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "K_shape": list(K.shape),
        "A_spectral_radius": float(np.max(np.abs(eig_a))),
        "A_cl_spectral_radius": float(np.max(np.abs(eig_acl))),
        "controllability_rank": rank,
        "controllability_full": bool(rank >= lifted_dim),
        "P_symmetric": sym_p,
        "P_trace": float(np.trace(P)),
        "P_frobenius": float(np.linalg.norm(P, "fro")),
        "P_min_eig_real": float(np.min(np.real(eig_p))),
        "P_max_eig_real": float(np.max(np.real(eig_p))),
        "P_num_negative_eig_real": int(np.sum(np.real(eig_p) < 0)),
    }


def one_step_metrics(model, X, X_next, U, state_labels: list[str] | None = None) -> dict:
    pred = np.asarray(model.predict(X, u=U))
    err = pred - X_next
    rmse = np.sqrt(np.mean(err**2, axis=0)).tolist()
    mae = np.mean(np.abs(err), axis=0).tolist()
    l2 = np.linalg.norm(err, axis=1)
    out = {
        "rmse_per_component": rmse,
        "mae_per_component": mae,
        "mean_l2_error": float(np.mean(l2)),
        "median_l2_error": float(np.median(l2)),
    }
    if state_labels is not None:
        out["state_labels"] = state_labels[: X.shape[1]]
    return out


def rollout_errors(
    model,
    X,
    X_next,
    U,
    horizon: int,
    n_starts: int,
    seed: int,
    episode_id: np.ndarray | None = None,
) -> dict:
    """
    Mean L2 error along an open-loop horizon using **recorded** actions from the dataset.

    If ``episode_id`` is provided (shape ``(N,)`` aligned with ``X``), start indices are drawn only
    where ``episode_id[s:s+horizon]`` is constant so rollouts do not cross episode boundaries.
    If ``episode_id`` is ``None``, sampling matches the original OfflineRL-style pool (legacy).

    Returns a dict with ``mean_l2_error_by_step``, ``episode_constrained``, ``n_valid_starts_pool``, etc.,
    or an ``error`` key if there are insufficient samples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    U = np.asarray(U)
    episode_id_arr = (
        np.asarray(episode_id, dtype=np.int64).reshape(-1) if episode_id is not None else None
    )

    if episode_id_arr is not None:
        if episode_id_arr.shape[0] != n:
            return {
                "error": "episode_id_length_mismatch",
                "horizon": horizon,
                "n_samples": int(n),
                "episode_id_len": int(episode_id_arr.shape[0]),
            }
        if n < horizon:
            return {"error": "insufficient_samples", "horizon": horizon, "n_samples": int(n)}
        valid_starts = [
            s
            for s in range(n - horizon + 1)
            if bool(np.all(episode_id_arr[s : s + horizon] == episode_id_arr[s]))
        ]
        if not valid_starts:
            return {
                "error": "no_valid_episode_starts",
                "horizon": horizon,
                "n_samples": int(n),
            }
        valid = np.asarray(valid_starts, dtype=np.int64)
        n_pick = min(n_starts, len(valid))
        idx = rng.choice(valid, size=n_pick, replace=False)
        episode_constrained = True
        n_valid_pool = int(len(valid))
    else:
        max_start = n - horizon - 1
        if max_start < 1:
            return {"error": "insufficient_samples", "horizon": horizon, "n_samples": int(n)}
        n_pick = min(n_starts, max_start)
        idx = rng.choice(max_start, size=n_pick, replace=False)
        episode_constrained = False
        n_valid_pool = int(max_start)

    per_h = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    for s in idx:
        x_pred = np.asarray(X[s], dtype=np.float64).copy()
        for h in range(horizon):
            if s + h >= len(U):
                break
            u_h = U[s + h]
            x_pred = np.asarray(
                model.predict(x_pred.reshape(1, -1), u=u_h.reshape(1, -1))
            ).reshape(-1)
            x_true = X_next[s + h]
            per_h[h] += float(np.linalg.norm(x_pred - x_true))
            counts[h] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_e = np.divide(per_h, np.maximum(counts, 1))
    out = {
        "horizon": horizon,
        "n_rollout_starts": int(n_pick),
        "mean_l2_error_by_step": mean_e.tolist(),
        "episode_constrained": episode_constrained,
        "n_valid_starts_pool": n_valid_pool,
    }
    return out
