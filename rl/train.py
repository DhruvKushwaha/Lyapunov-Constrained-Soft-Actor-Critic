#!/usr/bin/env python3
"""
Train SAC / LC-SAC / Lyapunov baselines on ``safe_control_gym`` tasks.

**Entry:** ``python train_rl.py`` (repo root) or ``python -m rl.train``.

**Algorithms**

- ``sac``          — Soft Actor-Critic.
- ``lcsac``        — Lyapunov-constrained SAC (needs EDMD + LQR assets).
- ``lcsac_mean``    — LC-SAC with mean violation aggregation (not CVaR) (needs EDMD + LQR assets).
- ``lyap_rs_sac``  — Lyapunov reward-shaping SAC (needs EDMD + LQR assets).

EDMD/Lyapunov algos: quadrotor_2D and cartpole presets.

**Outputs:** ``results/online/<algo>/<preset>/`` (plus ``train_summary.json``).

**Compare runs:** ``python -m rl.compare_summaries <train_summary.json | run_dir> ...``
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Register gym environments before make()
import safe_control_gym.envs  # noqa: F401
from safe_control_gym.controllers.sac.sac_utils import SACAgent, SACBuffer
from safe_control_gym.utils.registration import make

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_config_path(p: str | Path) -> Path:
    """Resolve YAML paths relative to repo root when not absolute."""
    path = Path(p)
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.is_file() else path


def path_for_summary(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


# (preset_name) -> registry id, env yaml (relative to repo root), sac yaml (relative)
PRESETS: dict[str, tuple[str, str, str]] = {
    "quadrotor_2d_track": (
        "quadrotor",
        "Params/Quadrotor_2D/env_track.yaml",
        "Params/Quadrotor_2D/sac.yaml",
    ),
    "quadrotor_2d_stab": (
        "quadrotor",
        "Params/Quadrotor_2D/env_stab.yaml",
        "Params/Quadrotor_2D/sac.yaml",
    ),
    "quadrotor_3d_stab": (
        "quadrotor",
        "Params/Quadrotor_3D/env_stab.yaml",
        "Params/Quadrotor_3D/sac.yaml",
    ),
    "quadrotor_3d_track_gym": (
        "quadrotor",
        "Params/Quadrotor_3D/env_track_gym.yaml",
        "Params/Quadrotor_3D/sac_gym.yaml",
    ),
    "quadrotor_3d_track": (
        "quadrotor",
        "Params/Quadrotor_3D/env_track_rl.yaml",
        "Params/Quadrotor_3D/sac.yaml",
    ),
    "cartpole_stab": (
        "cartpole",
        "Params/Cartpole/env_stab.yaml",
        "Params/Cartpole/sac.yaml",
    ),
    "cartpole_track": (
        "cartpole",
        "Params/Cartpole/env_track.yaml",
        "Params/Cartpole/sac.yaml",
    ),
}

LCSAC_ALLOWED_PRESETS = {"quadrotor_2d_track", "quadrotor_2d_stab", "cartpole_stab", "cartpole_track",
                         "quadrotor_3d_track", "quadrotor_3d_stab"}
LYAP_ALGOS_ALLOWED_PRESETS = {"quadrotor_2d_track", "quadrotor_2d_stab", "cartpole_stab", "cartpole_track",
                               "quadrotor_3d_track", "quadrotor_3d_stab"}

# Dimension of the physical tracking-error vector fed to the EDMD lift.
PRESET_STATE_ERROR_DIMS: dict[str, int] = {
    "quadrotor_2d_track": 6,
    "quadrotor_2d_stab":  6,
    "cartpole_stab":      4,
    "cartpole_track":     4,
    "quadrotor_3d_track": 12,
    "quadrotor_3d_stab":  12,
}

# Default EDMD asset paths (relative to repo root) per preset.
# Points to hyperparameter-tuned models (edmd.tune_* --retrain-best) which have
# ~5 orders-of-magnitude better P conditioning (log10(cond) ~5.4) than the
# original models (log10(cond) ~10-11).
PRESET_EDMD_ASSETS: dict[str, tuple[str, str]] = {
    "quadrotor_2d_track": ("results/edmd/quadrotor_2d_track/edmd_model.pkl",
                           "results/edmd/quadrotor_2d_track/lqr_matrices.npz"),
    "quadrotor_2d_stab":  ("results/edmd/quadrotor_2d_stab/edmd_model.pkl",
                           "results/edmd/quadrotor_2d_stab/lqr_matrices.npz"),
    "cartpole_stab":      ("results/edmd/cartpole/edmd_model.pkl",
                           "results/edmd/cartpole/lqr_matrices.npz"),
    "cartpole_track":     ("results/edmd/cartpole/edmd_model.pkl",
                           "results/edmd/cartpole/lqr_matrices.npz"),
    # 3D track: python -m edmd.collect_quad_3d && python -m edmd.tune_quad_3d
    "quadrotor_3d_track": ("results/edmd/quadrotor_3d/edmd_model.pkl",
                           "results/edmd/quadrotor_3d/lqr_matrices.npz"),
    # 3D stab:  python -m edmd.collect_quad_3d_stab && python -m edmd.tune_quad_3d_stab
    "quadrotor_3d_stab":  ("results/edmd/quadrotor_3d_stab/edmd_model.pkl",
                           "results/edmd/quadrotor_3d_stab/lqr_matrices.npz"),
}



def _get_edmd_action_bounds(env, env_config: dict, action_dim: int):
    """Return physical action bounds that match what the env actually applies.

    The EDMD B matrix is fitted on physical actions (e.g. motor thrusts in N).
    When normalized_rl_action_space=True the actor outputs [-1, 1], so the agent
    must rescale to the same physical space before applying B in the Lyapunov step.

    For quadrotor envs the denormalization is:
        u = (1 + norm_act_scale * a) * hover_thrust
    so the correct physical range is [(1-scale)*hover, (1+scale)*hover].

    Falls back to constraint bounds if the env does not expose hover_thrust.
    """
    if hasattr(env, "hover_thrust") and hasattr(env, "norm_act_scale"):
        ht = float(env.hover_thrust)
        ns = float(env.norm_act_scale)
        lo = np.full(action_dim, (1.0 - ns) * ht, dtype=np.float32)
        hi = np.full(action_dim, (1.0 + ns) * ht, dtype=np.float32)
        return lo, hi
    # fallback: read from constraint block
    for c in env_config.get("constraints", []):
        if c.get("constrained_variable") == "input":
            lb = c.get("lower_bounds")
            ub = c.get("upper_bounds")
            if lb and ub:
                return (
                    np.array(lb[:action_dim], dtype=np.float32),
                    np.array(ub[:action_dim], dtype=np.float32),
                )
    return None, None


def load_task_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "task_config" not in data:
        raise ValueError(f"{path} must contain a top-level 'task_config' key")
    return copy.deepcopy(data["task_config"])


def merge_sac_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    merged = copy.deepcopy(raw)
    merged.update(raw.get("algo_config", {}) or {})
    return merged


def apply_trajectory_override(env_config: dict, trajectory_type: str | None) -> None:
    if trajectory_type is None:
        return
    task = env_config.get("task", "")
    if task != "traj_tracking":
        print(
            f"Warning: --trajectory ignored (task is '{task}', not traj_tracking).",
            file=sys.stderr,
        )
        return
    if trajectory_type not in ("circle", "figure8", "square"):
        raise ValueError(
            f"trajectory_type must be one of circle, figure8, square; got {trajectory_type!r}"
        )
    env_config.setdefault("task_info", {})
    env_config["task_info"]["trajectory_type"] = trajectory_type
    print(f"Using trajectory_type override: {trajectory_type}")


def build_env(
    registry: str,
    task_config: dict,
    *,
    gui: bool = False,
) -> object:
    return make(registry, gui=gui, **task_config)


def default_output_dir(
    algo: str,
    preset: str | None,
    registry: str,
    env_stem: str | None = None,
) -> Path:
    if preset:
        tag = preset
    elif env_stem:
        tag = f"{registry}_{env_stem}"
    else:
        tag = registry
    return REPO_ROOT / "results" / "online" / algo / tag


def run_sac(args: argparse.Namespace) -> int:
    from SAC import (
        get_traj_suffix,
        plot_training_results,
        save_training_data,
        train_sac,
    )

    if args.env_yaml:
        if not args.registry or not args.sac_yaml:
            print(
                "With --env-yaml, both --registry and --sac-yaml are required.",
                file=sys.stderr,
            )
            return 2
        env_yaml = resolve_config_path(args.env_yaml)
        sac_yaml = resolve_config_path(args.sac_yaml)
        registry = args.registry
        preset_name = None
    else:
        preset_name = args.preset
        env_yaml = REPO_ROOT / PRESETS[preset_name][1]
        sac_yaml = REPO_ROOT / PRESETS[preset_name][2]
        registry = PRESETS[preset_name][0]

    env_config = load_task_config(env_yaml)
    if args.seed is not None:
        env_config["seed"] = int(args.seed)
    apply_trajectory_override(env_config, args.trajectory)

    sac_config = merge_sac_yaml(sac_yaml)
    if args.norm_obs is not None:
        sac_config["norm_obs"] = args.norm_obs
    use_entropy = sac_config.get("use_entropy_tuning", True)
    sac_config["use_entropy_tuning"] = use_entropy

    out = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir("sac", preset_name, registry, env_yaml.stem)
    )
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    print(f"Environment YAML: {env_yaml}")
    print(f"SAC YAML: {sac_yaml}")
    print(f"Registry: {registry}  ->  output: {out}")

    env = build_env(registry, env_config, gui=args.gui)

    hidden_dim = sac_config.get("hidden_dim", 256)
    activation = sac_config.get("activation", "relu")
    gamma = sac_config.get("gamma", 0.99)
    tau = sac_config.get("tau", 0.005)
    actor_lr = sac_config.get("actor_lr", 3e-4)
    critic_lr = sac_config.get("critic_lr", 3e-4)
    entropy_lr = sac_config.get("entropy_lr", 3e-4)
    init_temperature = sac_config.get("init_temperature", 0.2)
    train_interval = sac_config.get("train_interval", 1)
    train_batch_size = sac_config.get("train_batch_size", 256)
    max_env_steps = sac_config.get("max_env_steps", 200_000)
    warm_up_steps = sac_config.get("warm_up_steps", 5000)
    max_buffer_size = sac_config.get("max_buffer_size", 1_000_000)
    eval_batch_size = sac_config.get("eval_batch_size", 10)
    log_interval = sac_config.get("log_interval", 100)
    eval_interval = sac_config.get("eval_interval", 4000)

    agent = SACAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_dim=hidden_dim,
        gamma=gamma,
        tau=tau,
        use_entropy_tuning=use_entropy,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        entropy_lr=entropy_lr,
        activation=activation,
        init_temperature=init_temperature,
    )
    agent.to(device)

    if use_entropy:
        if not isinstance(agent.log_alpha, torch.nn.Parameter) or getattr(
            agent.log_alpha, "grad_fn", None
        ) is not None:
            val = (
                agent.log_alpha.detach().cpu().item()
                if isinstance(agent.log_alpha, torch.Tensor)
                else float(agent.log_alpha)
            )
            agent.log_alpha = torch.nn.Parameter(torch.tensor(val, device=device))
        agent.log_alpha.requires_grad = True
        agent.alpha_opt = torch.optim.Adam([agent.log_alpha], lr=entropy_lr)

    replay_buffer = SACBuffer(
        obs_space=env.observation_space,
        act_space=env.action_space,
        max_size=max_buffer_size,
        batch_size=train_batch_size,
    )

    traj_for_train = args.trajectory
    if env_config.get("task") != "traj_tracking":
        traj_for_train = None

    results = train_sac(
        agent=agent,
        env=env,
        replay_buffer=replay_buffer,
        max_steps=max_env_steps,
        warm_up_steps=warm_up_steps,
        train_interval=train_interval,
        train_batch_size=train_batch_size,
        eval_interval=eval_interval if eval_interval > 0 else 0,
        eval_batch_size=eval_batch_size,
        log_interval=log_interval if log_interval > 0 else 50,
        output_dir=out,
        norm_obs=bool(sac_config.get("norm_obs", False)),
        device=device,
        trajectory_type=traj_for_train,
        env_config=env_config,
        sac_config=sac_config,
    )

    suffix = get_traj_suffix(trajectory_type=traj_for_train, env_config=env_config)
    final_path = out / f"sac_model_final{suffix}.pth"
    torch.save(agent.state_dict(), str(final_path))
    print(f"Saved final model to {final_path}")

    # Always save lightweight training data for downstream comparison plots.
    np.save(out / f"episode_rewards{suffix}.npy", np.array(results["episode_rewards"]))
    if results.get("eval_rewards"):
        with open(out / f"eval_rewards{suffix}.json", "w", encoding="utf-8") as _f:
            json.dump(results["eval_rewards"], _f)

    if args.plots:
        plot_training_results(results, out, traj_for_train, env_config)
        save_training_data(
            results,
            out,
            traj_for_train,
            env_config,
            max_env_steps,
            warm_up_steps,
            train_interval,
            train_batch_size,
            hidden_dim,
            gamma,
            tau,
            actor_lr,
            critic_lr,
        )

    summary = {
        "algo": "sac",
        "preset": preset_name,
        "registry": registry,
        "env_yaml": path_for_summary(env_yaml),
        "sac_yaml": path_for_summary(sac_yaml),
        "best_eval_reward": results["best_eval_reward"],
        "num_episodes": len(results["episode_rewards"]),
    }
    with open(out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env.close()
    return 0


def run_lcsac(args: argparse.Namespace) -> int:
    import pickle

    from LC_SAC import LCSAC
    from LC_SAC_Train import train_lcsac
    from LC_SAC_Train import plot_training_results as plot_lcsac_results

    preset = args.preset
    if preset not in LCSAC_ALLOWED_PRESETS:
        print(
            f"lcsac only supports presets {LCSAC_ALLOWED_PRESETS}; got {preset!r}.",
            file=sys.stderr,
        )
        return 2

    env_yaml  = REPO_ROOT / PRESETS[preset][1]
    sac_yaml = (
        resolve_config_path(args.sac_yaml)
        if args.sac_yaml
        else REPO_ROOT / PRESETS[preset][2].replace("sac.yaml", "lcsac.yaml")
    )

    env_config = load_task_config(env_yaml)
    if args.seed is not None:
        env_config["seed"] = int(args.seed)
    is_tracking = env_config.get("task") == "traj_tracking"
    if is_tracking:
        apply_trajectory_override(env_config, args.trajectory)

    sac_config = merge_sac_yaml(sac_yaml)
    sac_config["use_entropy_tuning"] = sac_config.get("use_entropy_tuning", False)

    edmd_path = (
        resolve_config_path(args.edmd_model) if args.edmd_model
        else REPO_ROOT / PRESET_EDMD_ASSETS[preset][0]
    )
    lqr_path = (
        resolve_config_path(args.lqr_matrices) if args.lqr_matrices
        else REPO_ROOT / PRESET_EDMD_ASSETS[preset][1]
    )
    if not edmd_path.is_file() or not lqr_path.is_file():
        print(f"Missing EDMD assets:\n  {edmd_path}\n  {lqr_path}", file=sys.stderr)
        return 2

    with open(edmd_path, "rb") as f:
        edmd_model = pickle.load(f)
    mats = np.load(lqr_path)
    P_lifted = mats["P"]
    A = mats["A_lifted"]
    B = mats["B_lifted"]

    out = Path(args.output_dir) if args.output_dir else default_output_dir("lcsac", preset, PRESETS[preset][0])
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = build_env(PRESETS[preset][0], env_config, gui=args.gui)

    state_error_dim  = PRESET_STATE_ERROR_DIMS[preset]
    hidden_dim       = sac_config.get("hidden_dim",       128)
    gamma            = sac_config.get("gamma",            0.99)
    tau              = sac_config.get("tau",              0.005)
    actor_lr         = sac_config.get("actor_lr",         3e-4)
    critic_lr        = sac_config.get("critic_lr",        3e-4)
    entropy_lr       = sac_config.get("entropy_lr",       3e-4)
    init_temperature = sac_config.get("init_temperature", 0.2)
    use_entropy      = sac_config.get("use_entropy_tuning", False)
    train_interval   = sac_config.get("train_interval",   1)
    train_batch_size = sac_config.get("train_batch_size", 128)
    max_env_steps    = sac_config.get("max_env_steps",    300_000)
    warm_up_steps    = sac_config.get("warm_up_steps",    1000)
    max_buffer_size  = sac_config.get("max_buffer_size",  1_000_000)
    eval_batch_size  = sac_config.get("eval_batch_size",  10)
    log_interval     = sac_config.get("log_interval",     4000)
    eval_interval    = sac_config.get("eval_interval",    4000)
    lyap_ramp_steps  = sac_config.get("lyap_ramp_steps",  50000)
    lam_max          = sac_config.get("lam_max",          1.0)
    lam_lr           = sac_config.get("lam_lr",           1e-4)
    decay_rate       = sac_config.get("decay_rate",       1e-3)
    cvar_q           = sac_config.get("cvar_q",           0.9)

    action_range = [env.action_space.low.copy(), env.action_space.high.copy()]

    edmd_act_low, edmd_act_high = None, None
    if env_config.get("normalized_rl_action_space", False):
        edmd_act_low, edmd_act_high = _get_edmd_action_bounds(
            env, env_config, env.action_space.shape[0])

    agent = LCSAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_range=action_range,
        hidden_dim=hidden_dim,
        device=device,
        edmd_model=edmd_model,
        P_lifted=P_lifted,
        A=A,
        B=B,
        gamma=gamma,
        tau=tau,
        init_temperature=init_temperature,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        entropy_lr=entropy_lr,
        use_entropy_tuning=use_entropy,
        state_error_dim=state_error_dim,
        lyap_ramp_steps=lyap_ramp_steps,
        edmd_action_low=edmd_act_low,
        edmd_action_high=edmd_act_high,
        lam_max=lam_max,
        lam_lr=lam_lr,
        decay_rate=decay_rate,
        cvar_q=cvar_q,
    )

    from Modified_SAC_Buffer import SACBuffer as LCSACBuffer

    replay_buffer = LCSACBuffer(
        obs_space=env.observation_space,
        act_space=env.action_space,
        max_size=max_buffer_size,
        batch_size=train_batch_size,
        x_error_dim=state_error_dim,
    )

    results = train_lcsac(
        agent=agent,
        env=env,
        edmd_model=edmd_model,
        replay_buffer=replay_buffer,
        max_steps=max_env_steps,
        warm_up_steps=warm_up_steps,
        train_interval=train_interval,
        train_batch_size=train_batch_size,
        eval_interval=eval_interval if eval_interval > 0 else 0,
        eval_batch_size=eval_batch_size,
        log_interval=log_interval if log_interval > 0 else 50,
        output_dir=out,
        norm_obs=False,
        trajectory_type=args.trajectory if is_tracking else None,
        env_config=env_config,
        state_error_dim=state_error_dim,
    )

    from SAC import get_traj_suffix

    suffix = get_traj_suffix(trajectory_type=args.trajectory, env_config=env_config)
    final_path = out / f"lcsac_model_final{suffix}.pth"
    agent.save(str(final_path))
    print(f"Saved final model to {final_path}")

    # Always save lightweight training data for downstream comparison plots.
    np.save(out / f"episode_rewards{suffix}.npy", np.array(results["episode_rewards"]))
    if results.get("eval_rewards"):
        with open(out / f"eval_rewards{suffix}.json", "w", encoding="utf-8") as _f:
            json.dump(results["eval_rewards"], _f)
    lyap_entries = [
        {"step": l["step"], "lyap_loss": l["lyap_loss"]}
        for l in results.get("training_losses", [])
        if "lyap_loss" in l
    ]
    if lyap_entries:
        with open(out / f"lyap_loss{suffix}.json", "w", encoding="utf-8") as _f:
            json.dump(lyap_entries, _f)

    if args.plots:
        plot_lcsac_results(results, out, args.trajectory, env_config)

    summary = {
        "algo": "lcsac",
        "preset": preset,
        "env_yaml": path_for_summary(env_yaml),
        "sac_yaml": path_for_summary(sac_yaml),
        "edmd_model": path_for_summary(edmd_path),
        "lqr_matrices": path_for_summary(lqr_path),
        "best_eval_reward": results["best_eval_reward"],
        "num_episodes": len(results["episode_rewards"]),
        "lam_max": lam_max,
        "lam_lr": lam_lr,
        "decay_rate": decay_rate,
        "cvar_q": cvar_q,
    }
    with open(out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env.close()
    return 0


def run_lcsac_mean(args: argparse.Namespace) -> int:
    """Run LC-SAC-Meanunov (CLF-only) baseline."""
    import pickle
    from LCSAC_Mean import LCSACMeanAgent
    from LC_SAC_Train import train_lcsac

    preset = args.preset
    if preset not in LYAP_ALGOS_ALLOWED_PRESETS:
        print(f"lcsac_mean only supports presets {LYAP_ALGOS_ALLOWED_PRESETS}; got {preset!r}.",
              file=sys.stderr)
        return 2

    env_yaml  = REPO_ROOT / PRESETS[preset][1]
    sac_yaml  = (resolve_config_path(args.sac_yaml)
                 if args.sac_yaml
                 else REPO_ROOT / PRESETS[preset][2].replace("sac.yaml", "lcsac_mean.yaml"))

    env_config = load_task_config(env_yaml)
    if args.seed is not None:
        env_config["seed"] = int(args.seed)
    is_tracking = env_config.get("task") == "traj_tracking"
    if is_tracking:
        apply_trajectory_override(env_config, args.trajectory)

    cfg = merge_sac_yaml(sac_yaml)
    cfg["use_entropy_tuning"] = cfg.get("use_entropy_tuning", False)

    edmd_path = (resolve_config_path(args.edmd_model) if args.edmd_model
                 else REPO_ROOT / PRESET_EDMD_ASSETS[preset][0])
    lqr_path  = (resolve_config_path(args.lqr_matrices) if args.lqr_matrices
                 else REPO_ROOT / PRESET_EDMD_ASSETS[preset][1])
    if not edmd_path.is_file() or not lqr_path.is_file():
        print(f"Missing EDMD assets:\n  {edmd_path}\n  {lqr_path}", file=sys.stderr)
        return 2

    with open(edmd_path, "rb") as f:
        edmd_model = pickle.load(f)
    mats     = np.load(lqr_path)
    P_lifted = mats["P"]
    A        = mats["A_lifted"]
    B        = mats["B_lifted"]

    out = Path(args.output_dir) if args.output_dir else default_output_dir("lcsac_mean", preset, PRESETS[preset][0])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = build_env(PRESETS[preset][0], env_config, gui=args.gui)

    state_error_dim = PRESET_STATE_ERROR_DIMS[preset]

    edmd_act_low, edmd_act_high = None, None
    if env_config.get("normalized_rl_action_space", False):
        edmd_act_low, edmd_act_high = _get_edmd_action_bounds(
            env, env_config, env.action_space.shape[0])

    agent = LCSACMeanAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_range=[env.action_space.low.copy(), env.action_space.high.copy()],
        hidden_dim=cfg.get("hidden_dim", 128),
        device=device,
        edmd_model=edmd_model,
        P_lifted=P_lifted, A=A, B=B,
        gamma=cfg.get("gamma", 0.99),
        tau=cfg.get("tau", 0.005),
        init_temperature=cfg.get("init_temperature", 0.2),
        actor_lr=cfg.get("actor_lr", 3e-4),
        critic_lr=cfg.get("critic_lr", 3e-4),
        entropy_lr=cfg.get("entropy_lr", 3e-4),
        use_entropy_tuning=cfg.get("use_entropy_tuning", False),
        state_error_dim=state_error_dim,
        alpha_V=cfg.get("alpha_V", 0.01),
        lam_lr=cfg.get("lam_lr", 1e-4),
        lam_max=cfg.get("lam_max", 1.0),
        edmd_action_low=edmd_act_low,
        edmd_action_high=edmd_act_high,
    )

    from Modified_SAC_Buffer import SACBuffer as LCSACBuffer
    replay_buffer = LCSACBuffer(
        obs_space=env.observation_space,
        act_space=env.action_space,
        max_size=cfg.get("max_buffer_size", 1_000_000),
        batch_size=cfg.get("train_batch_size", 128),
        x_error_dim=state_error_dim,
    )

    results = train_lcsac(
        agent=agent, env=env, edmd_model=edmd_model,
        replay_buffer=replay_buffer,
        max_steps=cfg.get("max_env_steps", 300_000),
        warm_up_steps=cfg.get("warm_up_steps", 1000),
        train_interval=cfg.get("train_interval", 1),
        train_batch_size=cfg.get("train_batch_size", 128),
        eval_interval=cfg.get("eval_interval", 4000),
        eval_batch_size=cfg.get("eval_batch_size", 10),
        log_interval=cfg.get("log_interval", 4000),
        output_dir=out, norm_obs=False,
        trajectory_type=args.trajectory if is_tracking else None,
        env_config=env_config,
        state_error_dim=state_error_dim,
    )

    from SAC import get_traj_suffix
    suffix = get_traj_suffix(trajectory_type=args.trajectory, env_config=env_config)
    agent.save(str(out / f"lcsac_mean_model_final{suffix}.pth"))

    np.save(out / f"episode_rewards{suffix}.npy", np.array(results["episode_rewards"]))
    if results.get("eval_rewards"):
        import json as _json
        with open(out / f"eval_rewards{suffix}.json", "w") as _f:
            _json.dump(results["eval_rewards"], _f)
    lyap_entries = [{"step": l["step"], "lyap_loss": l["lyap_loss"]}
                    for l in results.get("training_losses", []) if "lyap_loss" in l]
    if lyap_entries:
        import json as _json
        with open(out / f"lyap_loss{suffix}.json", "w") as _f:
            _json.dump(lyap_entries, _f)

    import json as _json
    summary = {
        "algo": "lcsac_mean", "preset": preset,
        "env_yaml": path_for_summary(env_yaml), "sac_yaml": path_for_summary(sac_yaml),
        "edmd_model": path_for_summary(edmd_path), "lqr_matrices": path_for_summary(lqr_path),
        "best_eval_reward": results["best_eval_reward"],
        "num_episodes": len(results["episode_rewards"]),
        "lam_max": cfg.get("lam_max", 1.0),
        "lam_lr": cfg.get("lam_lr", 1e-4),
        "alpha_V": cfg.get("alpha_V", 0.01),
    }
    with open(out / "train_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)
    env.close()
    return 0


def run_lyap_rs_sac(args: argparse.Namespace) -> int:
    """Run Lyapunov Reward-Shaping SAC baseline."""
    import pickle
    import json as _json
    from Lyap_RS_SAC import calibrate_lyap_rs_weight, train_lyap_rs_sac

    preset = args.preset
    if preset not in LYAP_ALGOS_ALLOWED_PRESETS:
        print(f"lyap_rs_sac only supports presets {LYAP_ALGOS_ALLOWED_PRESETS}; got {preset!r}.",
              file=sys.stderr)
        return 2

    env_yaml = REPO_ROOT / PRESETS[preset][1]
    sac_yaml = (resolve_config_path(args.sac_yaml)
                if args.sac_yaml
                else REPO_ROOT / PRESETS[preset][2].replace("sac.yaml", "lyap_rs_sac.yaml"))

    env_config = load_task_config(env_yaml)
    if args.seed is not None:
        env_config["seed"] = int(args.seed)
    is_tracking = env_config.get("task") == "traj_tracking"
    if is_tracking:
        apply_trajectory_override(env_config, args.trajectory)

    cfg = merge_sac_yaml(sac_yaml)
    cfg["use_entropy_tuning"] = cfg.get("use_entropy_tuning", False)

    edmd_path = (resolve_config_path(args.edmd_model) if args.edmd_model
                 else REPO_ROOT / PRESET_EDMD_ASSETS[preset][0])
    lqr_path  = (resolve_config_path(args.lqr_matrices) if args.lqr_matrices
                 else REPO_ROOT / PRESET_EDMD_ASSETS[preset][1])
    if not edmd_path.is_file() or not lqr_path.is_file():
        print(f"Missing EDMD assets:\n  {edmd_path}\n  {lqr_path}", file=sys.stderr)
        return 2

    with open(edmd_path, "rb") as f:
        edmd_model = pickle.load(f)
    P_np = np.load(lqr_path)["P"]    # only P needed for V(z) = z^T P z

    out = Path(args.output_dir) if args.output_dir else default_output_dir("lyap_rs_sac", preset, PRESETS[preset][0])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = build_env(PRESETS[preset][0], env_config, gui=args.gui)
    state_error_dim = PRESET_STATE_ERROR_DIMS[preset]

    hidden_dim   = cfg.get("hidden_dim", 128)
    activation   = cfg.get("activation", "relu")
    gamma        = cfg.get("gamma", 0.99)
    tau          = cfg.get("tau", 0.005)
    actor_lr     = cfg.get("actor_lr", 3e-4)
    critic_lr    = cfg.get("critic_lr", 3e-4)
    entropy_lr   = cfg.get("entropy_lr", 3e-4)
    init_temp    = cfg.get("init_temperature", 0.2)
    use_entropy  = cfg.get("use_entropy_tuning", False)
    train_bs     = cfg.get("train_batch_size", 128)
    max_steps    = cfg.get("max_env_steps", 300_000)
    warm_up      = cfg.get("warm_up_steps", 1000)
    max_buf      = cfg.get("max_buffer_size", 1_000_000)
    lyap_w       = cfg.get("lyap_rs_weight", 0.0)

    agent = SACAgent(
        obs_space=env.observation_space, act_space=env.action_space,
        hidden_dim=hidden_dim, gamma=gamma, tau=tau,
        use_entropy_tuning=use_entropy,
        actor_lr=actor_lr, critic_lr=critic_lr, entropy_lr=entropy_lr,
        activation=activation, init_temperature=init_temp,
    )
    agent.to(device)
    if use_entropy:
        if not isinstance(agent.log_alpha, torch.nn.Parameter) or getattr(agent.log_alpha, "grad_fn", None):
            val = agent.log_alpha.detach().cpu().item() if isinstance(agent.log_alpha, torch.Tensor) else float(agent.log_alpha)
            agent.log_alpha = torch.nn.Parameter(torch.tensor(val, device=device))
        agent.log_alpha.requires_grad = True
        agent.alpha_opt = torch.optim.Adam([agent.log_alpha], lr=entropy_lr)

    replay_buffer = SACBuffer(
        obs_space=env.observation_space, act_space=env.action_space,
        max_size=max_buf, batch_size=train_bs,
    )

    # Auto-calibrate weight when lyap_rs_weight=0 (sentinel for "compute from data")
    if lyap_w == 0.0:
        lyap_w = calibrate_lyap_rs_weight(env, edmd_model, P_np, gamma=gamma,
                                          n_steps=cfg.get("calibrate_steps", 500),
                                          state_error_dim=state_error_dim)

    traj = args.trajectory if is_tracking else None
    results = train_lyap_rs_sac(
        agent=agent, env=env,
        edmd_model=edmd_model, P_np=P_np,
        replay_buffer=replay_buffer,
        max_steps=max_steps, warm_up_steps=warm_up,
        train_interval=cfg.get("train_interval", 1),
        train_batch_size=train_bs,
        lyap_rs_weight=lyap_w, gamma=gamma,
        eval_interval=cfg.get("eval_interval", 4000),
        eval_batch_size=cfg.get("eval_batch_size", 10),
        log_interval=cfg.get("log_interval", 4000),
        output_dir=out, device=device,
        trajectory_type=traj, env_config=env_config,
        sac_config=cfg,
        state_error_dim=state_error_dim,
    )

    from SAC import get_traj_suffix
    suffix = get_traj_suffix(trajectory_type=traj, env_config=env_config)
    torch.save(agent.state_dict(), str(out / f"lyap_rs_sac_model_final{suffix}.pth"))
    np.save(out / f"episode_rewards{suffix}.npy", np.array(results["episode_rewards"]))
    if results.get("eval_rewards"):
        with open(out / f"eval_rewards{suffix}.json", "w") as _f:
            _json.dump(results["eval_rewards"], _f)
    if results.get("lyap_entries"):
        with open(out / f"lyap_loss{suffix}.json", "w") as _f:
            _json.dump(results["lyap_entries"], _f)

    summary = {
        "algo": "lyap_rs_sac", "preset": preset,
        "env_yaml": path_for_summary(env_yaml), "sac_yaml": path_for_summary(sac_yaml),
        "edmd_model": path_for_summary(edmd_path), "lqr_matrices": path_for_summary(lqr_path),
        "lyap_rs_weight": lyap_w,
        "calibrate_steps": cfg.get("calibrate_steps", 500),
        "best_eval_reward": results["best_eval_reward"],
        "num_episodes": len(results["episode_rewards"]),
    }
    with open(out / "train_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)
    env.close()
    return 0



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SAC / LC-SAC / Lyapunov baselines on safe-control-gym envs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets:\n  " + "\n  ".join(sorted(PRESETS.keys())),
    )
    p.add_argument(
        "--algo",
        choices=("sac", "lcsac", "lcsac_mean", "lyap_rs_sac"),
        default="sac",
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Selects default env + SAC yaml (see Params/STRUCTURE.txt). Omit if using --env-yaml.",
    )
    p.add_argument(
        "--env-yaml",
        dest="env_yaml",
        default=None,
        help="Override task YAML (must contain task_config). Implies --registry if set.",
    )
    p.add_argument(
        "--sac-yaml",
        default=None,
        help="Override SAC hyperparameter YAML (algo + algo_config).",
    )
    p.add_argument(
        "--registry",
        choices=("quadrotor", "cartpole"),
        default=None,
        help="safe_control_gym registry id (required if --env-yaml is set without --preset defaults).",
    )
    p.add_argument(
        "--trajectory",
        choices=("circle", "figure8", "square"),
        default=None,
        help="Override trajectory_type for quadrotor traj_tracking tasks.",
    )
    p.add_argument("--output-dir", default=None, help="Where to save models and logs.")
    p.add_argument("--seed", type=int, default=None, help="Override task_config seed.")
    p.add_argument("--gui", action="store_true", help="PyBullet GUI.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip matplotlib training plots / numpy exports.",
    )
    p.add_argument(
        "--norm-obs",
        dest="norm_obs",
        action="store_true",
        help="(sac) Set norm_obs True in merged sac config.",
    )
    p.add_argument(
        "--no-norm-obs",
        dest="norm_obs",
        action="store_false",
        help="(sac) Set norm_obs False in merged sac config.",
    )
    p.set_defaults(plots=True, norm_obs=None)

    p.add_argument(
        "--edmd-model",
        default=None,
        help="(lcsac) Path to edmd_model_2D.pkl",
    )
    p.add_argument(
        "--lqr-matrices",
        default=None,
        help="(lcsac) Path to lqr_matrices_2D.npz",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.algo == "sac":
        if args.env_yaml is None and args.preset is None:
            args.preset = "quadrotor_2d_track"
    else:
        if args.preset is None:
            args.preset = "quadrotor_2d_track"
        if args.algo in ("lcsac", "lcsac_mean", "lyap_rs_sac"):
            if args.env_yaml or args.registry:
                print(f"{args.algo} does not support --env-yaml / --registry; use --preset.", file=sys.stderr)
                return 2
            if args.preset not in LYAP_ALGOS_ALLOWED_PRESETS:
                print(f"{args.algo} does not support preset {args.preset!r}. "
                      f"Supported: {sorted(LYAP_ALGOS_ALLOWED_PRESETS)}", file=sys.stderr)
                return 2
    if args.algo == "sac":
        return run_sac(args)
    if args.algo == "lcsac":
        return run_lcsac(args)
    if args.algo == "lcsac_mean":
        return run_lcsac_mean(args)
    if args.algo == "lyap_rs_sac":
        return run_lyap_rs_sac(args)
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
