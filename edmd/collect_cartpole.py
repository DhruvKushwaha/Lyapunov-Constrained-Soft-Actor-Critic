#!/usr/bin/env python3
"""
Collect Cartpole tracking-error data under LQR + action noise for EDMDc.

Saves transitions and ``episode_id`` to ``Saved_data/data_EDMD_cartpole.npz`` (or ``--output``).

Example::

  python -m edmd.collect_cartpole --num-episodes 40 --action-noise 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import yaml

import safe_control_gym.envs  # noqa: F401 — register envs
from safe_control_gym.utils.registration import make

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


def merge_lqr_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return dict(raw.get("algo_config") or {})


def goal_vector(env) -> np.ndarray:
    g = np.asarray(env.X_GOAL, dtype=np.float64).reshape(-1)
    if g.size != env.observation_space.shape[0]:
        raise RuntimeError(f"Unexpected X_GOAL shape {g.shape}, obs {env.observation_space.shape}")
    return g


def sanitize(data: dict) -> dict:
    te = data["tracking_error"]
    tn = data["tracking_error_next"]
    u = data["U"]
    mask = np.isfinite(te).all(axis=1) & np.isfinite(tn).all(axis=1) & np.isfinite(u).all(axis=1)
    if "episode_id" in data:
        eid = np.asarray(data["episode_id"], dtype=np.float64).reshape(-1)
        mask = mask & np.isfinite(eid)
    n_bad = int(np.sum(~mask))
    if n_bad:
        print(f"Removing {n_bad} non-finite rows.", file=sys.stderr)
    return {k: v[mask] for k, v in data.items()}


def collect_lqr_rollouts(
    task_config: dict,
    lqr_kwargs: dict,
    *,
    num_episodes: int,
    step_limit: int,
    action_noise_std: float,
    gui: bool,
    output_dir: str = "temp",
) -> dict:
    env_func = partial(make, "cartpole", **{**task_config, "gui": gui})

    ctrl = make(
        "lqr",
        env_func,
        output_dir=output_dir,
        training=False,
        **lqr_kwargs,
    )
    env = ctrl.env
    low = env.action_space.low
    high = env.action_space.high

    xs, us, xps = [], [], []
    tes, ten = [], []
    episode_ids = []

    g = goal_vector(env)

    episode_seq = 0
    for ep in range(num_episodes):
        episode_seq += 1
        obs, info = env.reset()
        if info is None:
            info = {}

        for _ in range(step_limit):
            u_lqr = np.asarray(ctrl.select_action(obs, info), dtype=np.float64).reshape(-1)
            if action_noise_std > 0:
                u = u_lqr + np.random.normal(0.0, action_noise_std, size=u_lqr.shape)
            else:
                u = u_lqr.copy()
            u = np.clip(u, low, high)

            next_obs, _r, done, next_info = env.step(u)
            if next_info is None:
                next_info = {}

            e = obs.astype(np.float64) - g
            e_n = next_obs.astype(np.float64) - g

            xs.append(obs)
            us.append(u)
            xps.append(next_obs)
            tes.append(e)
            ten.append(e_n)
            episode_ids.append(episode_seq)

            obs = next_obs
            info = next_info
            if done:
                break

        if (ep + 1) % max(1, num_episodes // 10) == 0 or ep == 0:
            print(f"  episode {ep + 1}/{num_episodes}, samples so far: {len(us)}")

    ctrl.close()

    return {
        "X": np.array(xs),
        "U": np.array(us),
        "X_prime": np.array(xps),
        "tracking_error": np.array(tes),
        "tracking_error_next": np.array(ten),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-yaml", type=str, default="Params/Cartpole/env_stab.yaml")
    p.add_argument("--lqr-yaml", type=str, default="Params/Cartpole/lqr_collect.yaml")
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--step-limit", type=int, default=2000)
    p.add_argument("--action-noise", type=float, default=0.02)
    p.add_argument("--output", type=str, default="results/edmd/data/cartpole.npz")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--no-sanitize", action="store_true")
    args = p.parse_args()

    env_yaml = resolve_path(args.env_yaml)
    lqr_yaml = resolve_path(args.lqr_yaml)
    if not env_yaml.is_file():
        print(f"Missing {env_yaml}", file=sys.stderr)
        return 2
    if not lqr_yaml.is_file():
        print(f"Missing {lqr_yaml}", file=sys.stderr)
        return 2

    task_config = load_task_config(env_yaml)
    if args.seed is not None:
        task_config["seed"] = int(args.seed)
    seed = task_config.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    lqr_kwargs = merge_lqr_yaml(lqr_yaml)

    print("Collecting cartpole EDMD data (LQR + noise)...")
    print(f"  env: {env_yaml}")
    print(f"  lqr: {lqr_kwargs}")

    data = collect_lqr_rollouts(
        task_config,
        lqr_kwargs,
        num_episodes=args.num_episodes,
        step_limit=args.step_limit,
        action_noise_std=args.action_noise,
        gui=args.gui,
        output_dir=str(REPO_ROOT / "RL_Model" / "temp_lqr_collect"),
    )

    if not args.no_sanitize:
        data = sanitize(data)

    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=data["X"],
        U=data["U"],
        X_prime=data["X_prime"],
        tracking_error=data["tracking_error"],
        tracking_error_next=data["tracking_error_next"],
        episode_id=data["episode_id"],
    )

    meta = {
        "controller": "lqr",
        "env_yaml": str(env_yaml),
        "lqr_yaml": str(lqr_yaml),
        "lqr_config": lqr_kwargs,
        "num_episodes": args.num_episodes,
        "step_limit": args.step_limit,
        "action_noise_std": args.action_noise,
        "output": str(out),
        "train_module": "edmd.train_cartpole",
        "shapes": {k: list(v.shape) for k, v in data.items() if isinstance(v, np.ndarray)},
        "num_samples": int(len(data["U"])),
    }
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {meta['num_samples']} transitions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
