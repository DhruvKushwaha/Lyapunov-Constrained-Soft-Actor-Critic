#!/usr/bin/env python3
"""
Collect 2D quadrotor PID rollouts for EDMDc.

Writes ``Saved_data/data_EDMD_2D.npz`` (or ``--output``) with ``tracking_error*``, ``U``, ``X*``, and
``episode_id`` (one integer per transition for episode-constrained rollout metrics in training).

Examples::

  python -m edmd.collect_quad_2d
  python -m edmd.collect_quad_2d --trajectories figure8,circle --num-episodes 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    cand = REPO_ROOT / path
    return cand if cand.is_file() else path


def load_task_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "task_config" not in data:
        raise ValueError(f"{path} must contain top-level 'task_config'")
    return dict(data["task_config"])


def load_pid_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_edmd_arrays(data: dict) -> dict:
    te = data["tracking_error"]
    tn = data["tracking_error_next"]
    u = data["U"]
    mask = np.isfinite(te).all(axis=1) & np.isfinite(tn).all(axis=1) & np.isfinite(u).all(axis=1)
    if "episode_id" in data:
        eid = np.asarray(data["episode_id"], dtype=np.float64).reshape(-1)
        mask = mask & np.isfinite(eid)
    if "offline_dataset" in data:
        od = data["offline_dataset"]
        mask = (
            mask
            & np.isfinite(od["observations"]).all(axis=1)
            & np.isfinite(od["next_observations"]).all(axis=1)
            & np.isfinite(od["actions"]).all(axis=1)
            & np.isfinite(od["rewards"]).reshape(-1)
            & np.isfinite(od["terminals"]).reshape(-1)
        )
    n_bad = int(np.sum(~mask))
    if n_bad:
        print(f"Removing {n_bad} rows with non-finite values.", file=sys.stderr)
    out = {}
    for k, v in data.items():
        if k == "offline_dataset":
            out[k] = {ok: ov[mask] for ok, ov in v.items()}
        else:
            out[k] = v[mask]
    return out


def main() -> int:
    from PID_controller_quadrotor import collect_edmd_data

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", choices=("pid",), default="pid")
    p.add_argument(
        "--env-yaml",
        type=str,
        default="Params/Quadrotor_2D/pid/env_track.yaml",
    )
    p.add_argument("--pid-yaml", type=str, default="Params/algorithms/pid_drone_gains.yaml")
    p.add_argument("--trajectories", type=str, default="figure8,circle,square")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--step-limit", type=int, default=500)
    p.add_argument("--action-noise", type=float, default=0.001)
    p.add_argument("--normalized-actions", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--output", type=str, default="results/edmd/data/quadrotor_2d_track.npz")
    p.add_argument(
        "--also-offline-npz",
        type=str,
        default=None,
        help=(
            "Also write OfflineRL-Kit MDP dataset (.npz) from the same PID+noise rollouts. "
            "Requires finite RL rewards (use env YAML with cost: rl_reward, e.g. Params/Quadrotor_2D/env_track.yaml)."
        ),
    )
    p.add_argument("--no-sanitize", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    env_yaml = resolve_path(args.env_yaml)
    pid_yaml = resolve_path(args.pid_yaml)
    if not env_yaml.is_file():
        print(f"Missing env yaml: {env_yaml}", file=sys.stderr)
        return 2
    if not pid_yaml.is_file():
        print(f"Missing pid yaml: {pid_yaml}", file=sys.stderr)
        return 2

    trajectories = [t.strip() for t in args.trajectories.split(",") if t.strip()]
    for t in trajectories:
        if t not in ("circle", "figure8", "square"):
            print(f"Unknown trajectory {t!r}; use circle, figure8, or square.", file=sys.stderr)
            return 2

    env_config = load_task_config(env_yaml)
    if args.seed is not None:
        env_config["seed"] = int(args.seed)

    pid_cfg = load_pid_config(pid_yaml)

    print("Collecting EDMD discrete data (2D PID baseline)...")
    print(f"  env: {env_yaml}")
    print(f"  pid: {pid_yaml}")
    print(f"  trajectories: {trajectories}, episodes each: {args.num_episodes}, step_limit: {args.step_limit}")

    also_off = bool(args.also_offline_npz)
    data = collect_edmd_data(
        env_config=env_config,
        pid_cfg=pid_cfg,
        trajectories=trajectories,
        num_episodes=args.num_episodes,
        step_limit=args.step_limit,
        action_noise_std=args.action_noise,
        normalized_action_space=args.normalized_actions,
        gui=args.gui,
        include_mdp_transitions=also_off,
    )

    if not args.no_sanitize:
        data = sanitize_edmd_arrays(data)

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        X=data["X"],
        U=data["U"],
        X_prime=data["X_prime"],
        tracking_error=data["tracking_error"],
        tracking_error_next=data["tracking_error_next"],
        episode_id=data["episode_id"],
    )

    meta = {
        "controller": args.controller,
        "env_yaml": str(env_yaml),
        "pid_yaml": str(pid_yaml),
        "trajectories": trajectories,
        "num_episodes": args.num_episodes,
        "step_limit": args.step_limit,
        "action_noise_std": args.action_noise,
        "normalized_action_space": args.normalized_actions,
        "output": str(out_path),
        "train_module": "edmd.train_quad_2d",
        "shapes": {
            "tracking_error": list(data["tracking_error"].shape),
            "tracking_error_next": list(data["tracking_error_next"].shape),
            "U": list(data["U"].shape),
        },
        "num_samples": int(len(data["U"])),
        "also_offline_npz": str(resolve_path(args.also_offline_npz)) if also_off else None,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {meta['num_samples']} transitions to {out_path}")
    print(f"Metadata: {meta_path}")

    if also_off:
        from rl.offline.transitions import assert_finite_mdp_rewards, save_dataset_npz

        try:
            assert_finite_mdp_rewards(data["offline_dataset"])
        except ValueError as e:
            print(str(e), file=sys.stderr)
            print(
                "Hint: use `--env-yaml Params/Quadrotor_2D/env_track.yaml` (rl_reward) for "
                "`--also-offline-npz`, or add rl_reward to your task YAML.",
                file=sys.stderr,
            )
            return 3

        off_path = resolve_path(args.also_offline_npz)
        off_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_npz(off_path, data["offline_dataset"])
        print(f"Saved offline MDP dataset ({len(data['offline_dataset']['rewards'])} transitions) to {off_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
