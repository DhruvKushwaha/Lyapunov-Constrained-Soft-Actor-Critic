#!/usr/bin/env python3
"""
Collect 2D quadrotor PID rollouts for EDMDc — **stabilization** task.

Writes ``Saved_data/data_EDMD_2D_stab.npz`` with ``tracking_error*``, ``U``, and
``episode_id`` arrays. The stabilization error is ``state[:6] - stab_goal`` where
``stab_goal`` is the constant hover reference from ``info['x_reference']``.

Examples::

  python -m edmd.collect_quad_2d_stab
  python -m edmd.collect_quad_2d_stab --num-episodes 40 --action-noise 0.005
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


def main() -> int:
    from PID_controller_quadrotor import make_quadrotor_2d_env
    from safe_control_gym.controllers.pid.pid import PID as PIDController

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-yaml",   type=str, default="Params/Quadrotor_2D/pid/env_stab.yaml")
    p.add_argument("--pid-yaml",   type=str, default="Params/algorithms/pid_drone_gains.yaml")
    p.add_argument("--num-episodes", type=int,   default=40)
    p.add_argument("--step-limit",   type=int,   default=250)
    p.add_argument("--action-noise", type=float, default=0.005)
    p.add_argument("--output",       type=str,   default="results/edmd/data/quadrotor_2d_stab.npz")
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--gui",          action="store_true")
    args = p.parse_args()

    env_yaml = resolve_path(args.env_yaml)
    pid_yaml = resolve_path(args.pid_yaml)
    for path, name in [(env_yaml, "env-yaml"), (pid_yaml, "pid-yaml")]:
        if not path.is_file():
            print(f"Missing {name}: {path}", file=sys.stderr)
            return 2

    with open(env_yaml, "r", encoding="utf-8") as f:
        env_config = yaml.safe_load(f)["task_config"]
    with open(pid_yaml, "r", encoding="utf-8") as f:
        pid_cfg = yaml.safe_load(f)

    if args.seed is not None:
        env_config["seed"] = int(args.seed)

    print("Collecting stabilization EDMD data (2D PID)...")
    print(f"  env:         {env_yaml}")
    print(f"  pid:         {pid_yaml}")
    print(f"  episodes:    {args.num_episodes}  step_limit: {args.step_limit}")
    print(f"  action noise: {args.action_noise}")

    env = make_quadrotor_2d_env(
        traj_type=None,
        gui=args.gui,
        normalized_action_space=False,
        env_config=env_config,
    )

    pid = PIDController(
        env_func=lambda: env,
        g=pid_cfg["g"],
        kf=pid_cfg["KF"],
        km=pid_cfg["KM"],
        p_coeff_for=np.array(pid_cfg["P_COEFF_FOR"]),
        i_coeff_for=np.array(pid_cfg["I_COEFF_FOR"]),
        d_coeff_for=np.array(pid_cfg["D_COEFF_FOR"]),
        p_coeff_tor=np.array(pid_cfg["P_COEFF_TOR"]),
        i_coeff_tor=np.array(pid_cfg["I_COEFF_TOR"]),
        d_coeff_tor=np.array(pid_cfg["D_COEFF_TOR"]),
        pwm2rpm_scale=pid_cfg["PWM2RPM_SCALE"],
        pwm2rpm_const=pid_cfg["PWM2RPM_CONST"],
        min_pwm=pid_cfg["MIN_PWM"],
        max_pwm=pid_cfg["MAX_PWM"],
    )

    data_X, data_U, data_X_prime = [], [], []
    data_te, data_te_next, data_eid = [], [], []

    ep_id = 0
    for ep in range(args.num_episodes):
        obs, info = env.reset()

        # Stabilization goal — constant (6,) reference from env
        stab_goal = np.asarray(info["x_reference"], dtype=np.float64).reshape(6)

        pid.reset_before_run(obs, info)

        for t in range(args.step_limit):
            state = obs[:6].copy()

            action = pid.select_action(obs, info)
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                action = np.clip(
                    np.nan_to_num(action, nan=0.0),
                    env.action_space.low, env.action_space.high,
                )

            # Add action noise for state-space coverage
            noisy_action = action + np.random.normal(0, args.action_noise, size=action.shape)
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)

            next_obs, _reward, done, next_info = env.step(noisy_action)
            next_state = next_obs[:6].copy()

            te      = state      - stab_goal
            te_next = next_state - stab_goal

            if np.isfinite(te).all() and np.isfinite(te_next).all() and np.isfinite(noisy_action).all():
                data_X.append(state)
                data_U.append(noisy_action)
                data_X_prime.append(next_state)
                data_te.append(te)
                data_te_next.append(te_next)
                data_eid.append(ep_id)

            obs = next_obs
            info = next_info
            if done:
                break

        ep_id += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep + 1}/{args.num_episodes}  total transitions so far: {len(data_U)}")

    env.close()

    if len(data_U) == 0:
        print("ERROR: no transitions collected!", file=sys.stderr)
        return 1

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    te_arr      = np.array(data_te,      dtype=np.float64)
    te_next_arr = np.array(data_te_next, dtype=np.float64)
    U_arr       = np.array(data_U,       dtype=np.float64)
    X_arr       = np.array(data_X,       dtype=np.float64)
    X_prime_arr = np.array(data_X_prime, dtype=np.float64)
    eid_arr     = np.array(data_eid,     dtype=np.int64)

    np.savez(
        out_path,
        tracking_error=te_arr,
        tracking_error_next=te_next_arr,
        U=U_arr,
        X=X_arr,
        X_prime=X_prime_arr,
        episode_id=eid_arr,
    )

    meta = {
        "task": "stabilization",
        "stab_goal": stab_goal.tolist(),
        "env_yaml": str(env_yaml),
        "pid_yaml": str(pid_yaml),
        "num_episodes": args.num_episodes,
        "step_limit": args.step_limit,
        "action_noise_std": args.action_noise,
        "output": str(out_path),
        "num_samples": int(len(data_U)),
        "train_module": "edmd.train_quad_2d_stab",
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(data_U)} transitions → {out_path}")
    print(f"  tracking_error shape: {te_arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
