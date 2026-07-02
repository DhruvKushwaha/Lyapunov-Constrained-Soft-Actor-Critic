#!/usr/bin/env python3
"""
Collect 3D quadrotor PID rollouts for EDMDc — **stabilization** task.

Rolls out PID from randomised initial states toward a constant hover goal and
stores tracking-error transitions (``state - stab_goal``) for EDMDc fitting.

Output defaults to ``results/edmd/data/quadrotor_3d_stab.npz``.

Examples::

  python -m edmd.collect_quad_3d_stab
  python -m edmd.collect_quad_3d_stab --num-episodes 60 --action-noise 0.005
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
    p.add_argument("--env-yaml",      type=str,   default="Params/Quadrotor_3D/pid/env_stab.yaml")
    p.add_argument("--pid-yaml",      type=str,   default="Params/algorithms/pid_drone_gains.yaml")
    p.add_argument("--num-episodes",  type=int,   default=50)
    p.add_argument("--step-limit",    type=int,   default=250)
    p.add_argument("--action-noise",  type=float, default=0.003)
    p.add_argument("--output",        type=str,   default="results/edmd/data/quadrotor_3d_stab.npz")
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--gui",           action="store_true")
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

    print("Collecting 3D stabilisation EDMD data (PID)...")
    print(f"  env:          {env_yaml}")
    print(f"  pid:          {pid_yaml}")
    print(f"  episodes:     {args.num_episodes}  step_limit: {args.step_limit}")
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

    n_state = int(env.state_space.shape[0])  # 12 for quad_type=3
    data_X, data_U, data_X_prime = [], [], []
    data_te, data_te_next, data_eid = [], [], []
    stab_goal = None

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        ep_goal = np.asarray(info["x_reference"], dtype=np.float64).reshape(n_state)
        if stab_goal is None:
            stab_goal = ep_goal  # constant across episodes; save for metadata
        pid.reset_before_run(obs, info)

        for _ in range(args.step_limit):
            state = obs[:n_state].copy()
            action = pid.select_action(obs, info)
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                action = np.clip(np.nan_to_num(action, nan=0.0),
                                 env.action_space.low, env.action_space.high)

            noisy_action = action + np.random.normal(0, args.action_noise, size=action.shape)
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)

            next_obs, _reward, done, next_info = env.step(noisy_action)
            next_state = next_obs[:n_state].copy()

            te      = state      - ep_goal
            te_next = next_state - ep_goal

            if np.isfinite(te).all() and np.isfinite(te_next).all() and np.isfinite(noisy_action).all():
                data_X.append(state)
                data_U.append(noisy_action)
                data_X_prime.append(next_state)
                data_te.append(te)
                data_te_next.append(te_next)
                data_eid.append(ep)

            obs, info = next_obs, next_info
            if done:
                break

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep + 1}/{args.num_episodes}  total transitions: {len(data_U)}")

    env.close()

    if not data_U:
        print("ERROR: no transitions collected!", file=sys.stderr)
        return 1

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        tracking_error=np.array(data_te,      dtype=np.float64),
        tracking_error_next=np.array(data_te_next, dtype=np.float64),
        U=np.array(data_U,       dtype=np.float64),
        X=np.array(data_X,       dtype=np.float64),
        X_prime=np.array(data_X_prime, dtype=np.float64),
        episode_id=np.array(data_eid,  dtype=np.int64),
    )

    meta = {
        "task": "stabilization",
        "stab_goal": stab_goal.tolist() if stab_goal is not None else None,
        "env_yaml": str(env_yaml),
        "pid_yaml": str(pid_yaml),
        "num_episodes": args.num_episodes,
        "step_limit": args.step_limit,
        "action_noise_std": args.action_noise,
        "output": str(out_path),
        "num_samples": len(data_U),
        "n_state": n_state,
        "train_module": "edmd.train_quad_3d_stab",
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(data_U)} transitions → {out_path}")
    print(f"  tracking_error shape: {np.array(data_te).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
