"""
Low-level **PID** control and **EDMD dataset collection** for quadrotors (2D and 3D task configs).

``make_quadrotor_2d_env`` builds a ``safe_control_gym`` ``Quadrotor`` from YAML task config.
``collect_edmd_data`` rolls out PID + action noise, stores ``X, U, X'``, tracking errors, and a per-step
``episode_id`` used by ``edmd.collect_quad_*`` for episode-aware metrics.

When run as a script, saves a demo trajectory and ``Saved_data/data_EDMD_2D.npz``.
"""

import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.benchmark_env import Task, Cost
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.controllers.pid.pid import PID as PIDController


def make_quadrotor_2d_env(
    traj_type="circle",
    gui=False,
    normalized_action_space=False,
    env_config=None
):
    """
    Create a ``Quadrotor`` environment (2D or 3D per task YAML, e.g. ``quad_type``).

    Args:
        traj_type: Trajectory type ('circle', 'figure8', 'square')
        gui: Whether to show PyBullet GUI
        normalized_action_space: Whether to use normalized action space
        env_config: Environment configuration dictionary
    """
    # Start with the loaded config
    config = env_config.copy()

    # Override trajectory type if specified
    if traj_type is not None:
        if traj_type not in ['circle', 'figure8', 'square']:
            raise ValueError(f"traj_type must be one of ['circle', 'figure8', 'square'], got '{traj_type}'")
        if 'task_info' not in config:
            config['task_info'] = {}
        config['task_info']['trajectory_type'] = traj_type

    # Convert quad_type integer to QuadType enum
    quad_type = QuadType(config["quad_type"])

    # Convert task string to Task enum
    task = Task(config["task"])

    # Convert cost string to Cost enum
    cost = Cost(config["cost"])

    # Extract init_state and inertial_prop
    init_state = config.get("init_state")
    inertial_prop = config.get("inertial_prop")

    # Build environment kwargs from config
    env_kwargs = {
        "seed": config.get("seed"),
        "quad_type": quad_type,
        "task": task,
        "physics": config.get("physics"),
        "task_info": config.get("task_info"),
        "cost": cost,
        "gui": gui,
        "normalized_rl_action_space": normalized_action_space,
        "pyb_freq": config.get("pyb_freq"),
        "ctrl_freq": config.get("ctrl_freq"),
        "episode_len_sec": config.get("episode_len_sec"),
        "randomized_init": config.get("randomized_init", True),
        "randomized_inertial_prop": config.get("randomized_inertial_prop", False),
        "init_state_randomization_info": config.get("init_state_randomization_info"),
        "obs_goal_horizon": config.get("obs_goal_horizon"),
        "rew_state_weight": config.get("rew_state_weight"),
        "rew_act_weight": config.get("rew_act_weight"),
        "rew_exponential": config.get("rew_exponential", True),
        "done_on_out_of_bound": config.get("done_on_out_of_bound", True),
        "info_mse_metric_state_weight": config.get("info_mse_metric_state_weight"),
        "constraints": config.get("constraints"),
        "done_on_violation": config.get("done_on_violation", False),
    }

    # Add optional parameters only if they exist
    if init_state is not None:
        env_kwargs["init_state"] = init_state
    if inertial_prop is not None:
        env_kwargs["inertial_prop"] = inertial_prop

    env = Quadrotor(**env_kwargs)
    return env


def run_single_episode(env, pid_controller, step_limit=500):
    """
    Run a single episode with PID control.

    Args:
        env: Quadrotor environment
        pid_controller: PID controller instance
        step_limit: Maximum number of steps

    Returns:
        states: Array of states
        actions: Array of actions
        rewards: Array of rewards
        x_ref_full: Reference trajectory
    """
    obs, info = env.reset()
    x_ref_full = info.get("x_reference", None)
    pid_controller.reset_before_run(obs, info)

    states, actions, rewards = [], [], []
    episode_reward = 0.0

    for t in range(step_limit):
        action = pid_controller.select_action(obs, info)

        # Check for NaN or invalid actions
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"Warning: Invalid action at step {t}: {action}")
            action = np.nan_to_num(action, nan=0.0, posinf=env.action_space.high[0], neginf=env.action_space.low[0])
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)

        # Check for NaN reward
        if np.isnan(reward) or np.isinf(reward):
            if t < 10 or t % 50 == 0:  # Print first 10 and then every 50th
                print(f"Warning: Invalid reward at step {t}: reward={reward}")
                print(f"  Action: {action}")
                if hasattr(env, 'state'):
                    print(f"  State: {env.state}")
                    if np.any(np.isnan(env.state)) or np.any(np.isinf(env.state)):
                        print(f"  State contains NaN/Inf!")
                if hasattr(env, 'ctrl_step_counter') and hasattr(env, 'X_GOAL'):
                    print(f"  Step counter: {env.ctrl_step_counter}, X_GOAL shape: {env.X_GOAL.shape}")
            reward = 0.0  # Replace NaN reward with 0

        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        # Only add valid rewards to episode_reward (NaN/inf already replaced with 0 above)
        # Double-check to ensure reward is valid before adding
        if not (np.isnan(reward) or np.isinf(reward)):
            episode_reward += reward
        else:
            # This shouldn't happen if the check above worked, but just in case
            episode_reward += 0.0

        if done:
            print(f"Episode terminated at step {t}: done={done}")
            if 'out_of_bounds' in info:
                print(f"  Out of bounds: {info.get('out_of_bounds', False)}")
            if 'goal_reached' in info:
                print(f"  Goal reached: {info.get('goal_reached', False)}")
            break

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    # Check for NaN in rewards array and fix
    if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
        nan_count = np.sum(np.isnan(rewards)) + np.sum(np.isinf(rewards))
        print(f"\nWarning: Found {nan_count} NaN/inf rewards in array")
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        episode_reward = np.sum(rewards)
        print(f"Corrected episode reward: {episode_reward:.2f}")

    print("\nEpisode reward:", episode_reward)
    print("Steps:", len(states))
    print("State shape:", states.shape)
    print("Action shape:", actions.shape)
    print(f"Reward stats: min={np.min(rewards):.4f}, max={np.max(rewards):.4f}, mean={np.mean(rewards):.4f}")

    return states, actions, rewards, x_ref_full


def plot_trajectory(states, x_ref_full, traj_type="figure8"):
    """
    Plot the trajectory comparison between reference and actual.

    Args:
        states: Array of actual states
        x_ref_full: Reference trajectory
        traj_type: Type of trajectory (for title)
    """
    # Plot trajectory (2D quadrotor moves in XZ plane)
    if x_ref_full is not None:
        # For 2D: state is [x, x_dot, z, z_dot, theta, theta_dot]
        # Extract positions: x (index 0) and z (index 2)
        ref_pos = x_ref_full[:, [0, 2]]
    else:
        ref_pos = None

    # Extract actual positions from states
    # States are observations which may be 24D, but actual state is first 6D
    if states.shape[1] > 6:
        actual_pos = states[:, [0, 2]]  # Extract x and z from first 6 dimensions
    else:
        actual_pos = states[:, [0, 2]]  # x and z positions

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    if ref_pos is not None:
        ax.plot(ref_pos[:, 0], ref_pos[:, 1],
                'r--', linewidth=2, label='Reference Trajectory')

    ax.plot(actual_pos[:, 0], actual_pos[:, 1],
            'b-', linewidth=2, label='PID Quadrotor Trajectory')

    ax.set_title(f"PID: Reference vs Quadrotor Trajectory (XZ plane) - {traj_type} Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def collect_edmd_data(
    env_config,
    pid_cfg,
    trajectories=['figure8', 'circle', 'square'],
    num_episodes=20,
    step_limit=500,
    action_noise_std=0.001,
    normalized_action_space=False,
    gui=False,
    *,
    include_mdp_transitions=False,
    return_edmd=True,
    reward_scale=1.0,
):
    """
    Roll out PID control (2D or 3D quad from ``env_config``) and return a transition dataset for EDMD.

    State slices use ``env.state_space`` dimensionality. Each stored row shares one ``episode_id`` integer
    for all steps within the same environment episode (used by ``edmd.metrics.rollout_errors``).

    Args:
        env_config: ``task_config``-style dict for ``Quadrotor``.
        pid_cfg: YAML-loaded PID gains (thrust/torque, PWM limits, etc.).
        trajectories: Trajectory names passed to ``make_quadrotor_2d_env``.
        num_episodes: Episodes per trajectory.
        step_limit: Max steps per episode.
        action_noise_std: Gaussian noise std on actions after PID.
        normalized_action_space: Forwarded to the env constructor.
        gui: PyBullet GUI flag.
        include_mdp_transitions: If True, also build an OfflineRL-Kit-style transition dict (same rollouts).
        return_edmd: If False, skip EDMD arrays (use with ``include_mdp_transitions`` for offline-only runs).
        reward_scale: Multiplier on env reward when ``include_mdp_transitions`` is True.

    Returns:
        If ``return_edmd``, keys ``X``, ``U``, ``X_prime``, ``tracking_error``, ``tracking_error_next``,
        ``episode_id`` (aligned rows). If ``include_mdp_transitions``, key ``offline_dataset`` with
        ``observations``, ``next_observations``, ``actions``, ``rewards``, ``terminals``.
    """
    if not return_edmd and not include_mdp_transitions:
        raise ValueError("At least one of return_edmd or include_mdp_transitions must be True.")
    # Storage for EDMD (X, U, X_next)
    data_X = []
    data_U = []
    data_X_prime = []
    data_tracking_error = []
    data_tracking_error_next = []
    data_episode_id = []
    data_obs = []
    data_next_obs = []
    data_rew = []
    data_term = []
    episode_seq = 0

    # Collect data for each trajectory
    for traj in trajectories:
        env = make_quadrotor_2d_env(
            gui=gui,
            traj_type=traj,
            normalized_action_space=normalized_action_space,
            env_config=env_config,
        )
        n_state = int(env.state_space.shape[0])

        # Create PID controller for this environment
        def env_func():
            return env

        pid_controller_traj = PIDController(
            env_func=env_func,
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

        # Run the environment for specified number of episodes
        for episode in range(num_episodes):
            obs, info = env.reset()
            x_ref_full = info.get("x_reference", None)
            if return_edmd and x_ref_full is None:
                raise ValueError(
                    "collect_edmd_data needs info['x_reference'] for tracking-error labels; "
                    "use task traj_tracking (see Quadrotor reset info)."
                )
            pid_controller_traj.reset_before_run(obs, info)
            episode_seq += 1
            episode_uid = episode_seq

            # Run the environment for step limit
            for step in range(step_limit):
                if obs.shape[0] > n_state:
                    X = obs[:n_state]
                else:
                    X = obs

                action = pid_controller_traj.select_action(obs, info)
                action += np.random.normal(0, action_noise_std, action.shape)
                action_low = env.action_space.low
                action_high = env.action_space.high
                noisy_action = np.clip(action, action_low, action_high)

                next_obs, reward, done, info = env.step(noisy_action)

                if next_obs.shape[0] > n_state:
                    X_next = next_obs[:n_state]
                else:
                    X_next = next_obs

                if return_edmd:
                    data_X.append(X)
                    data_U.append(noisy_action)
                    data_X_prime.append(X_next)
                    data_episode_id.append(episode_uid)
                elif include_mdp_transitions:
                    data_U.append(noisy_action)

                if return_edmd:
                    # Align references with LC_SAC_Train.train_lcsac: pre-step state uses ref index
                    # (episode_length / ctrl index before this step). After step(), info["current_step"] is
                    # env.ctrl_step_counter = number of completed control steps, so pre-step index is current_step - 1.
                    current_step = int(info.get("current_step", 0))
                    idx_pre = max(0, current_step - 1)
                    if idx_pre >= x_ref_full.shape[0]:
                        idx_pre = x_ref_full.shape[0] - 1
                    idx_next = min(current_step, x_ref_full.shape[0] - 1)

                    x_ref_current = x_ref_full[idx_pre]
                    x_ref_next = x_ref_full[idx_next]

                    tracking_error = X - x_ref_current
                    tracking_error_next = X_next - x_ref_next

                    data_tracking_error.append(tracking_error)
                    data_tracking_error_next.append(tracking_error_next)

                if include_mdp_transitions:
                    data_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                    data_next_obs.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
                    data_rew.append(float(reward) * float(reward_scale))
                    data_term.append(1.0 if bool(done) else 0.0)

                obs = next_obs

                if done:
                    break

            if episode % 5 == 0:
                print(f"Completed episode {episode+1}/{num_episodes} for trajectory {traj}")

        # Close the environment for this trajectory
        env.close()
        print(f"Completed data collection for trajectory: {traj}")

    print("\nData collection completed for all trajectories.")

    out: dict = {}

    if return_edmd:
        out["X"] = np.array(data_X)
        out["U"] = np.array(data_U)
        out["X_prime"] = np.array(data_X_prime)
        out["tracking_error"] = np.array(data_tracking_error)
        out["tracking_error_next"] = np.array(data_tracking_error_next)
        out["episode_id"] = np.asarray(data_episode_id, dtype=np.int64)

    if include_mdp_transitions:
        n_mdp = len(data_obs)
        if n_mdp == 0:
            raise RuntimeError("include_mdp_transitions set but no transitions were collected.")
        out["offline_dataset"] = {
            "observations": np.asarray(data_obs, dtype=np.float32),
            "next_observations": np.asarray(data_next_obs, dtype=np.float32),
            "actions": np.asarray(data_U, dtype=np.float32),
            "rewards": np.asarray(data_rew, dtype=np.float32).reshape(-1),
            "terminals": np.asarray(data_term, dtype=np.float32).reshape(-1),
        }
    return out


def main():
    """Main execution function."""
    # Load PID configuration
    PID_CONFIG_PATH = Path("./Params/Controllers/pid.yaml")
    assert PID_CONFIG_PATH.exists(), f"Missing {PID_CONFIG_PATH}"

    with open(PID_CONFIG_PATH, "r") as f:
        pid_cfg = yaml.safe_load(f)

    print("Loaded PID config from pid.yaml:")
    print(pid_cfg)

    # Load environment configuration from YAML (2D config)
    ENV_CONFIG_PATH = Path("./Params/Quadrotor_2D/PID/quadrotor_2D_track.yaml")
    assert ENV_CONFIG_PATH.exists(), f"Missing {ENV_CONFIG_PATH}"

    with open(ENV_CONFIG_PATH, "r") as f:
        env_config = yaml.safe_load(f)["task_config"]

    print("\nLoaded environment config from quadrotor_2D_track.yaml")

    # ============================================================================
    # Part 1: Run a single episode and plot trajectory
    # ============================================================================
    print("\n" + "="*70)
    print("Part 1: Running single episode with PID control")
    print("="*70)

    # Select the trajectory type
    traj = "figure8"
    env = make_quadrotor_2d_env(gui=False, traj_type=traj, env_config=env_config)

    def env_func():
        return env

    # Create PID controller (it will reset the environment internally during initialization)
    pid_controller = PIDController(
        env_func=env_func,
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

    print("\nPID controller created.")

    # Get initial observation and info after PID controller initialization
    obs, info = env.reset()
    print("\nObs shape:", obs.shape)
    print("Action space:", env.action_space)

    # Run single episode
    step_limit = 500
    states, actions, rewards, x_ref_full = run_single_episode(
        env, pid_controller, step_limit
    )

    # Plot trajectory
    print("\nPlotting trajectory...")
    plot_trajectory(states, x_ref_full, traj)

    # ============================================================================
    # Part 2: Collect data for EDMD
    # ============================================================================
    print("\n" + "="*70)
    print("Part 2: Collecting data for EDMD training")
    print("="*70)

    TRAJECTORIES = ['figure8', 'circle', 'square']
    edmd_data = collect_edmd_data(
        env_config=env_config,
        pid_cfg=pid_cfg,
        trajectories=TRAJECTORIES,
        num_episodes=20,
        step_limit=step_limit
    )

    # Create save_data directory
    SAVE_DATA_DIR = Path("Saved_data")
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSave data directory: {SAVE_DATA_DIR.absolute()}")

    # Save the data for EDMD in save_data folder
    data_file_path = SAVE_DATA_DIR / 'data_EDMD_2D.npz'
    np.savez(
        data_file_path,
        X=edmd_data["X"],
        U=edmd_data["U"],
        X_prime=edmd_data["X_prime"],
        tracking_error=edmd_data["tracking_error"],
        tracking_error_next=edmd_data["tracking_error_next"],
        episode_id=edmd_data["episode_id"],
    )

    # Check shape and type of data
    print(f"Data X shape: {edmd_data['X'].shape}, type: {edmd_data['X'].dtype}")
    print(f"Data U shape: {edmd_data['U'].shape}, type: {edmd_data['U'].dtype}")
    print(f"Data X_prime shape: {edmd_data['X_prime'].shape}, type: {edmd_data['X_prime'].dtype}")
    print(f"Data tracking_error shape: {edmd_data['tracking_error'].shape}, type: {edmd_data['tracking_error'].dtype}")
    print(f"Data tracking_error_next shape: {edmd_data['tracking_error_next'].shape}, type: {edmd_data['tracking_error_next'].dtype}")
    print(f"\nSaved data to {data_file_path}")

    # Close the environment
    env.close()
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
