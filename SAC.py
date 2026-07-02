"""
Baseline **SAC** training utilities for the 2D quadrotor (trajectory tracking).

Exports ``train_sac``, ``make_quadrotor_2d_env``, plotting/saving helpers, and evaluation entry points used by
``rl.train`` (``--algo sac``). Running this file directly still launches a full training run;
for presets and 3D/cartpole, prefer ``python train_rl.py --algo sac`` from the repo root.
"""

import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import copy
import json
from pathlib import Path
import yaml

from safe_control_gym.controllers.sac.sac_utils import SACAgent, SACBuffer

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.benchmark_env import Task, Cost
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType

# Clear cuda cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def get_traj_suffix(trajectory_type=None, env_config=None):
    """
    Get trajectory type suffix for file naming.

    Args:
        trajectory_type: Explicit trajectory type (from TRAJECTORY_TYPE)
        env_config: Environment config dict (to extract trajectory_type if not provided)

    Returns:
        String suffix like '_circle', '_figure8', '_square', or '' if None
    """
    if trajectory_type is None:
        if env_config is not None:
            task_info = env_config.get('task_info', {})
            trajectory_type = task_info.get('trajectory_type')

    if trajectory_type is None:
        return ''
    else:
        return f'_{trajectory_type}'


def make_quadrotor_2d_env(gui=False, override_config=None, trajectory_type=None, env_config=None):
    """
    Create a Quadrotor 2D environment.

    Args:
        gui: Whether to show PyBullet GUI
        override_config: Optional dict to override specific config values
        trajectory_type: Optional trajectory type to use. Options: 'circle', 'figure8', 'square'
        env_config: Base environment configuration dictionary
    """
    # Start with the loaded config
    config = env_config.copy()

    # Override trajectory type if specified
    if trajectory_type is not None:
        if trajectory_type not in ['circle', 'figure8', 'square']:
            raise ValueError(f"trajectory_type must be one of ['circle', 'figure8', 'square'], got '{trajectory_type}'")
        if 'task_info' not in config:
            config['task_info'] = {}
        config['task_info']['trajectory_type'] = trajectory_type
        print(f"✓ Using trajectory type: {trajectory_type}")

    # Apply any overrides
    if override_config:
        config.update(override_config)

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
        "normalized_rl_action_space": config.get("normalized_rl_action_space", False),
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


def evaluate_agent(agent, env, num_episodes=10, device=None, obs_normalizer=None):
    """
    Evaluate agent on environment.

    Args:
        agent: Trained agent
        env: Environment
        num_episodes: Number of evaluation episodes
        device: Device for agent operations
        obs_normalizer: Optional observation normalizer

    Returns:
        Average reward over episodes
    """
    agent.eval()
    total_rewards = []

    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.ac.act(torch.FloatTensor(state).unsqueeze(0).to(device), deterministic=True)
                if isinstance(action, np.ndarray) and action.ndim > 0:
                    action = action[0]  # Remove batch dimension if present
            state, reward, done, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def train_sac(agent, env, replay_buffer, max_steps, warm_up_steps, train_interval,
              train_batch_size, eval_interval=0, eval_batch_size=10, log_interval=0,
              output_dir=None, obs_normalizer=None, norm_obs=False, device=None,
              trajectory_type=None, env_config=None, sac_config=None):
    """
    Train SAC agent on environment.

    Args:
        agent: SAC agent
        env: Environment
        replay_buffer: Replay buffer
        max_steps: Maximum training steps
        warm_up_steps: Warm-up steps before training
        train_interval: Training interval
        train_batch_size: Training batch size
        eval_interval: Evaluation interval (0 to disable)
        eval_batch_size: Number of episodes for evaluation
        log_interval: Logging interval
        output_dir: Output directory for saving models
        obs_normalizer: Observation normalizer
        norm_obs: Whether observation normalization is enabled
        device: Device for agent operations
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
        sac_config: SAC configuration dictionary

    Returns:
        Dictionary with training results
    """
    if output_dir is None:
        output_dir = Path("results/online/sac/quadrotor_2d_track/standalone")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []

    # Evaluation metrics
    eval_rewards = []
    best_eval_reward = float('-inf')

    # Initialize environment
    state, info = env.reset()

    episode_reward = 0
    episode_length = 0
    total_steps = 0

    print(f"Starting training for {max_steps} steps (warm-up: {warm_up_steps})")

    # Set agent to training mode
    agent.train()

    while total_steps < max_steps:
        # Select action
        if total_steps < warm_up_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.ac.act(torch.FloatTensor(state).unsqueeze(0).to(device), deterministic=False)
                if isinstance(action, np.ndarray) and action.ndim > 0:
                    action = action[0]  # Remove batch dimension if present

        # Step environment
        next_state, reward, done, info = env.step(action)

        # Handle time truncation# Store transition
        mask = 1.0 - done.astype(np.float32) if isinstance(done, np.ndarray) else (1.0 - float(done))

        is_timeout = (done
                      and hasattr(env, 'ctrl_step_counter')
                      and hasattr(env, 'CTRL_STEPS')
                      and env.ctrl_step_counter >= env.CTRL_STEPS)
        true_mask = 1.0 if is_timeout else mask
        true_next_state = next_state

        # Ensure observations are properly shaped for SACBuffer
        obs_to_push = state.reshape(1, -1) if state.ndim == 1 else state
        next_obs_to_push = true_next_state.reshape(1, -1) if true_next_state.ndim == 1 else true_next_state
        act_to_push = action.reshape(1, -1) if action.ndim == 1 else action

        # Store transition
        transition_dict = {
            'obs': obs_to_push,
            'act': act_to_push,
            'rew': np.array([reward]),
            'next_obs': next_obs_to_push,
            'mask': np.array([true_mask]),
        }

        replay_buffer.push(transition_dict)

        state = next_state
        episode_reward += reward
        episode_length += 1
        total_steps += 1

        # Train agent — gated by train_interval to match LC-SAC update frequency exactly
        if (total_steps > warm_up_steps and len(replay_buffer) >= train_batch_size
                and (total_steps % train_interval == 0)):
            batch = replay_buffer.sample(batch_size=train_batch_size, device=device)
            losses = agent.update(batch)
            training_losses.append({
                'step': total_steps,
                'critic_loss': losses.get('critic_loss', 0.0),
                'actor_loss': losses.get('policy_loss', 0.0),
                'entropy_loss': losses.get('entropy_loss', 0.0),
                'alpha': agent.alpha.item()
            })

        # Handle episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Logging
            if log_interval > 0 and total_steps % log_interval == 0:
                n_recent = min(50, len(episode_rewards))
                recent_rewards = episode_rewards[-n_recent:]
                recent_lengths = episode_lengths[-n_recent:]
                print(f"Episode {len(episode_rewards)}, "
                      f"Steps: {total_steps}, "
                      f"Reward: {episode_reward:.2f} (avg: {np.mean(recent_rewards):.2f}), "
                      f"Length: {episode_length} (avg: {np.mean(recent_lengths):.1f})")

            # Reset environment
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0

        # Evaluation
        if eval_interval > 0 and total_steps % eval_interval == 0 and total_steps > warm_up_steps:
            eval_reward = evaluate_agent(agent, env, eval_batch_size, device, obs_normalizer if norm_obs else None)
            eval_rewards.append({
                'step': total_steps,
                'reward': eval_reward
            })

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                if sac_config and sac_config.get("eval_save_best", False):
                    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)
                    best_model_path = output_dir / f"best_sac_model{traj_suffix}.pth"
                    torch.save(agent.state_dict(), str(best_model_path))
                    print(f"Saved best model to {best_model_path}")

            print(f"Evaluation at step {total_steps}: Avg reward = {eval_reward:.2f} "
                  f"(Best: {best_eval_reward:.2f})")
            # evaluate_agent resets env multiple times; reset here so the next training
            # step starts from a clean env state instead of a stale one.
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            # Restore training mode so entropy tuning gradients are enabled.
            agent.train()

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
        'eval_rewards': eval_rewards,
        'best_eval_reward': best_eval_reward
    }


def plot_training_results(training_results, output_dir, trajectory_type=None, env_config=None):
    """
    Plot and save training results.

    Args:
        training_results: Dictionary with training results
        output_dir: Output directory
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
    """
    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(training_results['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Moving average of episode rewards
    if len(training_results['episode_rewards']) > 20:
        window = min(100, len(training_results['episode_rewards']) // 10)
        moving_avg = np.convolve(training_results['episode_rewards'],
                                np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)

    # Episode lengths
    axes[1, 0].plot(training_results['episode_lengths'])
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].grid(True)

    # Evaluation rewards
    if training_results['eval_rewards']:
        eval_steps = [r['step'] for r in training_results['eval_rewards']]
        eval_rewards = [r['reward'] for r in training_results['eval_rewards']]
        axes[1, 1].plot(eval_steps, eval_rewards, 'o-')
        axes[1, 1].set_title('Evaluation Rewards')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No evaluation data',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Evaluation Rewards')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"training_plots{traj_suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training plots to {plot_path}")

    plt.show()


def save_training_data(training_results, output_dir, trajectory_type=None, env_config=None,
                      max_env_steps=0, warm_up_steps=0, train_interval=0, train_batch_size=0,
                      hidden_dim=0, gamma=0, tau=0, actor_lr=0, critic_lr=0):
    """
    Save training data to files.

    Args:
        training_results: Dictionary with training results
        output_dir: Output directory
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
        Additional config parameters for summary
    """
    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)

    print("\nSaving training data...")

    # Save episode rewards
    episode_rewards_path = output_dir / f"episode_rewards{traj_suffix}.npy"
    np.save(episode_rewards_path, np.array(training_results['episode_rewards']))
    print(f"Saved episode rewards to {episode_rewards_path}")

    # Save episode lengths
    episode_lengths_path = output_dir / f"episode_lengths{traj_suffix}.npy"
    np.save(episode_lengths_path, np.array(training_results['episode_lengths']))
    print(f"Saved episode lengths to {episode_lengths_path}")

    # Save training losses
    if training_results['training_losses']:
        training_losses_path = output_dir / f"training_losses{traj_suffix}.json"
        with open(training_losses_path, 'w') as f:
            json.dump(training_results['training_losses'], f, indent=2)
        print(f"Saved training losses to {training_losses_path}")

        # Also save as numpy array
        if training_results['training_losses'] and 'critic_loss' in training_results['training_losses'][0]:
            actor_loss_key = 'actor_loss' if 'actor_loss' in training_results['training_losses'][0] else 'policy_loss'
            loss_array = np.array([[l['step'], l['critic_loss'], l.get(actor_loss_key, 0.0),
                                   l.get('entropy_loss', 0.0), l.get('alpha', 0.2)]
                                  for l in training_results['training_losses']])
        else:
            loss_array = np.array([[l['step'], l.get('critic_1_loss', 0.0), l.get('critic_2_loss', 0.0),
                                   l.get('actor_loss', l.get('policy_loss', 0.0)), l.get('alpha', 0.2)]
                                  for l in training_results['training_losses']])
        np.save(output_dir / f"training_losses{traj_suffix}.npy", loss_array)

    # Save evaluation rewards
    if training_results['eval_rewards']:
        eval_rewards_path = output_dir / f"eval_rewards{traj_suffix}.json"
        with open(eval_rewards_path, 'w') as f:
            json.dump(training_results['eval_rewards'], f, indent=2)
        print(f"Saved evaluation rewards to {eval_rewards_path}")

        # Also save as numpy array
        eval_array = np.array([[r['step'], r['reward']] for r in training_results['eval_rewards']])
        np.save(output_dir / f"eval_rewards{traj_suffix}.npy", eval_array)

    # Save summary statistics
    summary = {
        'total_episodes': len(training_results['episode_rewards']),
        'best_eval_reward': training_results['best_eval_reward'],
        'final_10_episode_avg_reward': float(np.mean(training_results['episode_rewards'][-10:])
                                            if len(training_results['episode_rewards']) >= 10 else 0),
        'max_episode_reward': float(np.max(training_results['episode_rewards'])
                                  if training_results['episode_rewards'] else 0),
        'min_episode_reward': float(np.min(training_results['episode_rewards'])
                                  if training_results['episode_rewards'] else 0),
        'mean_episode_reward': float(np.mean(training_results['episode_rewards'])
                                   if training_results['episode_rewards'] else 0),
        'mean_episode_length': float(np.mean(training_results['episode_lengths'])
                                    if training_results['episode_lengths'] else 0),
        'config': {
            'max_env_steps': max_env_steps,
            'warm_up_steps': warm_up_steps,
            'train_interval': train_interval,
            'train_batch_size': train_batch_size,
            'hidden_dim': hidden_dim,
            'gamma': gamma,
            'tau': tau,
            'alpha': 0.02,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
        }
    }

    summary_path = output_dir / f"training_summary{traj_suffix}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")

    print("\nAll training data saved successfully!")


def evaluate_trajectory(agent, env, output_dir, trajectory_type=None, env_config=None,
                       hidden_dim=128, gamma=0.99, tau=0.005, actor_lr=0.0005, critic_lr=0.001,
                       entropy_lr=0.0001, use_entropy_tuning=True, activation='relu', device=None):
    """
    Evaluate trained agent and plot trajectory.

    Args:
        agent: Trained agent (or None to load from file)
        env: Environment
        output_dir: Output directory
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
        Additional parameters for agent initialization if loading from file
        device: Device for agent operations
    """
    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)

    # Load model if agent not provided
    if agent is None:
        model_path = output_dir / f"best_sac_model{traj_suffix}.pth"
        if not model_path.exists():
            model_path = output_dir / f"sac_model_final{traj_suffix}.pth"

        if model_path.exists():
            print(f"Loading model from {model_path}")
            eval_agent = SACAgent(
                obs_space=env.observation_space,
                act_space=env.action_space,
                hidden_dim=hidden_dim,
                gamma=gamma,
                tau=tau,
                use_entropy_tuning=use_entropy_tuning,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                entropy_lr=entropy_lr,
                activation=activation
            )
            eval_agent.to(device)
            state_dict = torch.load(str(model_path), weights_only=False)
            eval_agent.load_state_dict(state_dict)
            # Fix log_alpha after loading
            if 'log_alpha' in state_dict:
                loaded_log_alpha = state_dict['log_alpha']
                log_alpha_value = loaded_log_alpha.item() if hasattr(loaded_log_alpha, 'item') else float(loaded_log_alpha)
                eval_agent.log_alpha = torch.tensor(log_alpha_value, device=device, requires_grad=use_entropy_tuning)
                if use_entropy_tuning:
                    eval_agent.alpha_opt = torch.optim.Adam([eval_agent.log_alpha], lr=entropy_lr)
            eval_agent.eval()
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {model_path}")
            return
    else:
        eval_agent = agent
        eval_agent.eval()

    # Run evaluation episode
    print("\nRunning evaluation episode...")
    state, info = env.reset()

    agent_states = []
    agent_trajectory = []

    # Store initial state
    agent_states.append(env.state.copy())
    agent_trajectory.append({
        'state': env.state.copy(),
        'action': None,
        'reward': 0,
        'step': 0
    })

    done = False
    step_count = 0

    print(f"Initial state: {env.state}")

    while not done:
        with torch.no_grad():
            action = eval_agent.ac.act(torch.FloatTensor(state).unsqueeze(0).to(device), deterministic=True)
            if isinstance(action, np.ndarray) and action.ndim > 0:
                action = action[0]

        next_state, reward, done, info = env.step(action)

        step_count += 1

        agent_states.append(env.state.copy())
        agent_trajectory.append({
            'state': env.state.copy(),
            'action': action.copy(),
            'reward': reward,
            'step': step_count
        })

        if done:
            print(f"Episode terminated at step {step_count}")

        state = next_state

        if step_count > 10000:
            print("Warning: Exceeded 10000 steps, breaking loop")
            break

    agent_states = np.array(agent_states)
    print(f"Episode completed: {step_count} steps, collected {len(agent_states)} states")

    # Extract reference trajectory
    reference_trajectory = env.X_GOAL
    print(f"Reference trajectory shape: {reference_trajectory.shape}")
    print(f"Agent trajectory shape: {agent_states.shape}")

    # Extract positions (x, z)
    ref_positions = reference_trajectory[:, [0, 2]]
    agent_positions = agent_states[:, [0, 2]]

    # Truncate to same length
    min_len = min(len(ref_positions), len(agent_positions))
    ref_positions = ref_positions[:min_len]
    agent_positions = agent_positions[:min_len]

    # Create time steps
    ctrl_freq = env_config.get('ctrl_freq', 50) if env_config else 50
    dt = 1.0 / ctrl_freq
    time_steps = np.arange(len(agent_positions)) * dt

    # Plot trajectory
    fig = plt.figure(figsize=(10, 8))

    # XZ 2D trajectory plot
    ax1 = fig.add_subplot(211)
    ax1.plot(ref_positions[:, 0], ref_positions[:, 1],
             'b-', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1],
             'r--', linewidth=2, label='SAC Agent', alpha=0.7)
    ax1.scatter(ref_positions[0, 0], ref_positions[0, 1],
               c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(ref_positions[-1, 0], ref_positions[-1, 1],
               c='red', s=100, marker='*', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('XZ Trajectory: Reference vs SAC Agent (2D Quadrotor)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')

    # Calculate tracking error metrics
    position_error = agent_positions - ref_positions
    euclidean_error = np.linalg.norm(position_error, axis=1)
    mean_error = np.mean(euclidean_error)
    max_error = np.max(euclidean_error)
    rmse = np.sqrt(np.mean(euclidean_error**2))

    # Plot tracking error over time
    ax2 = fig.add_subplot(212)
    ax2.plot(time_steps, euclidean_error, 'g-', linewidth=2, alpha=0.7)
    ax2.axhline(y=mean_error, color='r', linestyle='--', linewidth=2,
               label=f'Mean Error: {mean_error:.4f} m')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Euclidean Error (m)')
    ax2.set_title('Position Tracking Error Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    trajectory_plot_path = output_dir / f"evaluation_trajectory{traj_suffix}.png"
    plt.savefig(trajectory_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved trajectory plot to {trajectory_plot_path}")

    plt.show()

    # Print metrics
    print(f"\n--- Tracking Error Metrics ---")
    print(f"Mean Euclidean Error: {mean_error:.4f} m")
    print(f"Max Euclidean Error: {max_error:.4f} m")
    print(f"RMSE: {rmse:.4f} m")

    # Save trajectory data
    trajectory_data = {
        'reference_positions': ref_positions.tolist(),
        'agent_positions': agent_positions.tolist(),
        'position_error': position_error.tolist(),
        'euclidean_error': euclidean_error.tolist(),
        'time_steps': time_steps.tolist(),
        'metrics': {
            'mean_error': float(mean_error),
            'max_error': float(max_error),
            'rmse': float(rmse)
        }
    }

    trajectory_data_path = output_dir / f"trajectory_data{traj_suffix}.json"
    with open(trajectory_data_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"Saved trajectory data to {trajectory_data_path}")


def plot_additional_analysis(training_results, output_dir, trajectory_type=None, env_config=None,
                            agent_trajectory=None, agent_states=None, reference_trajectory=None):
    """
    Plot additional analysis including losses, alpha, actions, and state components.

    Args:
        training_results: Dictionary with training results
        output_dir: Output directory
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
        agent_trajectory: Agent trajectory from evaluation
        agent_states: Agent states from evaluation
        reference_trajectory: Reference trajectory
    """
    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)

    # Try to load data if not provided
    episode_rewards = training_results.get('episode_rewards')
    training_losses = training_results.get('training_losses')
    eval_rewards = training_results.get('eval_rewards')

    if episode_rewards is None:
        try:
            ep_path = output_dir / f'episode_rewards{traj_suffix}.npy'
            if ep_path.exists():
                episode_rewards = list(np.load(ep_path))
        except Exception:
            pass

    if training_losses is None:
        try:
            loss_path = output_dir / f'training_losses{traj_suffix}.npy'
            if loss_path.exists():
                training_losses = np.load(loss_path)
        except Exception:
            pass

    if eval_rewards is None:
        try:
            eval_path = output_dir / f'eval_rewards{traj_suffix}.npy'
            if eval_path.exists():
                eval_rewards = np.load(eval_path)
        except Exception:
            pass

    # Create plots
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs = axs.ravel()

    # 1) Episode rewards
    if episode_rewards is not None and len(episode_rewards) > 0:
        axs[0].plot(episode_rewards, color='C0')
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].grid(True)
    else:
        axs[0].text(0.5, 0.5, 'No episode rewards available', ha='center', va='center')

    # 2) Moving average rewards
    if episode_rewards is not None and len(episode_rewards) > 5:
        w = max(1, min(200, len(episode_rewards)//10))
        mv = np.convolve(episode_rewards, np.ones(w)/w, mode='valid')
        axs[1].plot(mv, color='C1')
        axs[1].set_title(f'Moving Average Rewards (window={w})')
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, 'Insufficient reward data for moving average', ha='center', va='center')

    # 3) Training losses
    if training_losses is not None and len(training_losses) > 0:
        try:
            if isinstance(training_losses, list):
                steps = [l.get('step', i) for i, l in enumerate(training_losses)]
                critic = [l.get('critic_loss', l.get('critic_1_loss', np.nan)) for l in training_losses]
                actor = [l.get('actor_loss', l.get('policy_loss', np.nan)) for l in training_losses]
                entropy = [l.get('entropy_loss', np.nan) for l in training_losses]
                alpha = [l.get('alpha', np.nan) for l in training_losses]
            else:
                arr = np.array(training_losses)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    steps = arr[:, 0]
                    critic = arr[:, 1]
                    actor = arr[:, 2]
                    entropy = arr[:, 3] if arr.shape[1] > 3 else np.full_like(critic, np.nan)
                    alpha = arr[:, 4] if arr.shape[1] > 4 else np.full_like(critic, np.nan)
                else:
                    raise ValueError('Unknown training_losses array format')
        except Exception:
            steps = []
            critic = []
            actor = []
            entropy = []
            alpha = []

        if len(critic) > 0:
            axs[2].plot(steps, critic, label='critic_loss', color='C2')
        if len(actor) > 0:
            axs[2].plot(steps, actor, label='actor_loss', color='C3')
        axs[2].set_title('Training Losses')
        axs[2].legend()
        axs[2].grid(True)
    else:
        axs[2].text(0.5, 0.5, 'No training loss data', ha='center', va='center')

    # 4) Alpha (temperature) over time
    if training_losses is not None and len(training_losses) > 0 and 'alpha' in locals():
        try:
            if len(alpha) > 0:
                axs[3].plot(steps, alpha, color='C4')
                axs[3].set_title('Alpha (entropy temperature)')
                axs[3].grid(True)
            else:
                axs[3].text(0.5, 0.5, 'No alpha data', ha='center', va='center')
        except Exception:
            axs[3].text(0.5, 0.5, 'Alpha plotting failed', ha='center', va='center')
    else:
        axs[3].text(0.5, 0.5, 'No alpha data', ha='center', va='center')

    # 5) Action distribution
    if agent_trajectory is not None and len(agent_trajectory) > 1:
        acts = [a['action'] for a in agent_trajectory if a.get('action') is not None]
        try:
            acts_arr = np.array(acts)
            for i in range(min(3, acts_arr.shape[1])):
                axs[4].hist(acts_arr[:, i], bins=30, alpha=0.6, label=f'act_{i}')
            axs[4].set_title('Action Distributions (eval)')
            axs[4].legend()
        except Exception:
            axs[4].text(0.5, 0.5, 'Failed to compute action histogram', ha='center', va='center')
    else:
        axs[4].text(0.5, 0.5, 'No evaluation actions available', ha='center', va='center')

    # 6) State components
    try:
        if agent_states is not None and reference_trajectory is not None:
            ag = np.array(agent_states)
            ref = np.array(reference_trajectory)
            min_len = min(len(ag), len(ref))
            t = np.arange(min_len) * (1.0 / env_config.get('ctrl_freq', 50) if env_config else 50)
            axs[5].plot(t, ref[:min_len, 0], label='ref_x', color='C0')
            axs[5].plot(t, ag[:min_len, 0], '--', label='agent_x', color='C0', alpha=0.7)
            axs[5].plot(t, ref[:min_len, 2], label='ref_z', color='C1')
            axs[5].plot(t, ag[:min_len, 2], '--', label='agent_z', color='C1', alpha=0.7)
            axs[5].set_title('State components (x,z) Agent vs Ref')
            axs[5].legend()
            axs[5].grid(True)
        else:
            axs[5].text(0.5, 0.5, 'State or reference trajectory not available', ha='center', va='center')
    except Exception:
        axs[5].text(0.5, 0.5, 'Failed to plot state components', ha='center', va='center')

    plt.tight_layout()
    save_path = output_dir / f'additional_plots{traj_suffix}.png'
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved additional plots to {save_path}')
    except Exception as e:
        print('Could not save additional plots:', e)

    plt.show()


def main():
    """Main execution function."""
    # Configuration
    OUTPUT_DIR = Path("results/online/sac/quadrotor_2d_track/standalone")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # Load environment configuration
    ENV_CONFIG_PATH = Path("./Params/Quadrotor_2D/env_track.yaml")
    assert ENV_CONFIG_PATH.exists(), f"Missing {ENV_CONFIG_PATH}"

    with open(ENV_CONFIG_PATH, "r") as f:
        env_config = yaml.safe_load(f)["task_config"]

    print("Loaded environment config from Params/Quadrotor_2D/env_track.yaml")

    # Load SAC algorithm configuration
    SAC_CONFIG_PATH = Path("./Params/Quadrotor_2D/sac.yaml")
    assert SAC_CONFIG_PATH.exists(), f"Missing {SAC_CONFIG_PATH}"

    with open(SAC_CONFIG_PATH, "r") as f:
        sac_config_base = yaml.safe_load(f)

    with open(SAC_CONFIG_PATH, "r") as f:
        sac_config_override = yaml.safe_load(f).get("algo_config", {})

    # Merge configs
    sac_config = copy.deepcopy(sac_config_base)
    sac_config.update(sac_config_override)

    print("Loaded SAC algorithm config")

    # Trajectory type configuration
    TRAJECTORY_TYPE = 'circle'  # Options: None, 'circle', 'figure8', 'square'

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config settings
    sac_config['use_entropy_tuning'] = True
    sac_config['norm_obs'] = False

    # Extract config values
    HIDDEN_DIM = sac_config.get("hidden_dim", 128)
    ACTIVATION = sac_config.get("activation", "relu")
    NORM_OBS = sac_config.get("norm_obs", False)
    NORM_REWARD = sac_config.get("norm_reward", False)
    GAMMA = sac_config.get("gamma", 0.99)
    TAU = sac_config.get("tau", 0.005)
    USE_ENTROPY_TUNING = sac_config.get("use_entropy_tuning", True)
    TARGET_ENTROPY = sac_config.get("target_entropy")
    TRAIN_INTERVAL = sac_config.get("train_interval", 10)
    TRAIN_BATCH_SIZE = sac_config.get("train_batch_size", 256)
    ACTOR_LR = sac_config.get("actor_lr", 0.0005)
    CRITIC_LR = sac_config.get("critic_lr", 0.001)
    ENTROPY_LR = sac_config.get("entropy_lr", 0.0001)
    MAX_ENV_STEPS = sac_config.get("max_env_steps", 300000)
    WARM_UP_STEPS = sac_config.get("warm_up_steps", 500)
    INIT_TEMPERATURE = sac_config.get("init_temperature", 0.2)
    MAX_BUFFER_SIZE = sac_config.get("max_buffer_size", 1000000)
    EVAL_BATCH_SIZE = sac_config.get("eval_batch_size", 10)
    LOG_INTERVAL = sac_config.get("log_interval", 100)
    EVAL_INTERVAL = sac_config.get("eval_interval", 4000)

    print(f"\nSAC Config: hidden_dim={HIDDEN_DIM}, batch_size={TRAIN_BATCH_SIZE}, max_steps={MAX_ENV_STEPS}")
    if USE_ENTROPY_TUNING:
        print("✓ Using automatic entropy tuning")

    # Initialize environment and agent
    env = make_quadrotor_2d_env(gui=False, trajectory_type=TRAJECTORY_TYPE, env_config=env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [env.action_space.low.copy(), env.action_space.high.copy()]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action range: [{action_range[0]}, {action_range[1]}]")

    # Initialize SAC agent
    agent = SACAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        tau=TAU,
        use_entropy_tuning=USE_ENTROPY_TUNING,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        entropy_lr=ENTROPY_LR,
        activation=ACTIVATION,
        init_temperature=INIT_TEMPERATURE
    )
    agent.to(device)

    # Ensure `log_alpha` is a leaf `Parameter` when using entropy tuning (fixes requires_grad error)
    if USE_ENTROPY_TUNING:
        try:
            if hasattr(agent, 'log_alpha'):
                # If not a Parameter or has grad_fn (non-leaf), replace with a leaf Parameter
                if not isinstance(agent.log_alpha, torch.nn.Parameter) or getattr(agent.log_alpha, 'grad_fn', None) is not None:
                    val = agent.log_alpha.detach().cpu().item() if isinstance(agent.log_alpha, torch.Tensor) else float(agent.log_alpha)
                    agent.log_alpha = torch.nn.Parameter(torch.tensor(val, device=device))
                agent.log_alpha.requires_grad = True
                # (Re)create optimizer for log_alpha if entropy tuning is used
                agent.alpha_opt = torch.optim.Adam([agent.log_alpha], lr=ENTROPY_LR)
        except Exception as _e:
            print('Warning: could not reinitialize agent.log_alpha:', _e)

    # Initialize replay buffer
    replay_buffer = SACBuffer(
        obs_space=env.observation_space,
        act_space=env.action_space,
        max_size=MAX_BUFFER_SIZE,
        batch_size=TRAIN_BATCH_SIZE
    )

    print("Environment and agent initialized successfully!")

    # Training
    print("Starting SAC training...")
    print(f"Config: max_steps={MAX_ENV_STEPS}, warm_up={WARM_UP_STEPS}, "
          f"train_interval={TRAIN_INTERVAL}, eval_interval={EVAL_INTERVAL}")

    training_results = train_sac(
        agent=agent,
        env=env,
        replay_buffer=replay_buffer,
        max_steps=MAX_ENV_STEPS,
        warm_up_steps=WARM_UP_STEPS,
        train_interval=TRAIN_INTERVAL,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_interval=EVAL_INTERVAL if EVAL_INTERVAL > 0 else 0,
        eval_batch_size=EVAL_BATCH_SIZE,
        log_interval=LOG_INTERVAL if LOG_INTERVAL > 0 else 50,
        output_dir=OUTPUT_DIR,
        norm_obs=False,
        device=device,
        trajectory_type=TRAJECTORY_TYPE,
        env_config=env_config,
        sac_config=sac_config
    )

    print("\nTraining completed!")
    print(f"Total episodes: {len(training_results['episode_rewards'])}")
    print(f"Best evaluation reward: {training_results['best_eval_reward']:.2f}")
    print(f"Average final 10 episode rewards: {np.mean(training_results['episode_rewards'][-10:]):.2f}")

    # Plot and save training results
    plot_training_results(training_results, OUTPUT_DIR, TRAJECTORY_TYPE, env_config)
    save_training_data(training_results, OUTPUT_DIR, TRAJECTORY_TYPE, env_config,
                      MAX_ENV_STEPS, WARM_UP_STEPS, TRAIN_INTERVAL, TRAIN_BATCH_SIZE,
                      HIDDEN_DIM, GAMMA, TAU, ACTOR_LR, CRITIC_LR)

    # Save final model
    traj_suffix = get_traj_suffix(trajectory_type=TRAJECTORY_TYPE, env_config=env_config)
    final_model_path = OUTPUT_DIR / f"sac_model_final{traj_suffix}.pth"
    torch.save(agent.state_dict(), str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    # Evaluate trajectory
    eval_env = make_quadrotor_2d_env(
        gui=False,
        override_config={'randomized_init': False},
        trajectory_type=TRAJECTORY_TYPE,
        env_config=env_config
    )

    # Run evaluation to get trajectory data
    state, info = eval_env.reset()
    agent_trajectory = []
    agent_states = []
    agent_states.append(eval_env.state.copy())
    agent_trajectory.append({
        'state': eval_env.state.copy(),
        'action': None,
        'reward': 0,
        'step': 0
    })

    done = False
    step_count = 0

    while not done:
        with torch.no_grad():
            action = agent.ac.act(torch.FloatTensor(state).unsqueeze(0).to(device), deterministic=True)
            if isinstance(action, np.ndarray) and action.ndim > 0:
                action = action[0]

        next_state, reward, done, info = eval_env.step(action)
        step_count += 1

        agent_states.append(eval_env.state.copy())
        agent_trajectory.append({
            'state': eval_env.state.copy(),
            'action': action.copy(),
            'reward': reward,
            'step': step_count
        })

        state = next_state

        if step_count > 10000:
            break

    agent_states = np.array(agent_states)
    reference_trajectory = eval_env.X_GOAL

    # Evaluate and plot trajectory
    evaluate_trajectory(
        agent=agent,
        env=eval_env,
        output_dir=OUTPUT_DIR,
        trajectory_type=TRAJECTORY_TYPE,
        env_config=env_config,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        entropy_lr=ENTROPY_LR,
        use_entropy_tuning=USE_ENTROPY_TUNING,
        activation=ACTIVATION,
        device=device
    )

    # Plot additional analysis
    plot_additional_analysis(
        training_results,
        OUTPUT_DIR,
        TRAJECTORY_TYPE,
        env_config,
        agent_trajectory,
        agent_states,
        reference_trajectory
    )

    # Close environments
    env.close()
    eval_env.close()
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
