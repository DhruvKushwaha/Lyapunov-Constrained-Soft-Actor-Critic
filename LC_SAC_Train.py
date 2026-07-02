"""
**LC-SAC** training loop for 2D quadrotor trajectory tracking.

Loads EDMD + LQR artifacts, constructs ``LCSAC`` from ``LC_SAC``, and trains with ``Modified_SAC_Buffer.SACBuffer``.
Imported by ``rl.train`` (``--algo lcsac``). Direct execution runs one training
job; use ``train_rl.py`` for a unified CLI and consistent ``results/online/`` outputs.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import json
import pickle
from pathlib import Path
import yaml

from LC_SAC import LCSAC
from Modified_SAC_Buffer import SACBuffer
from SAC import make_quadrotor_2d_env, get_traj_suffix

# Clear cuda cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# get_traj_suffix and make_quadrotor_2d_env are imported from SAC (single source of truth)


def evaluate_agent(agent, env, num_episodes=10, obs_normalizer=None):
    """
    Evaluate agent on environment.

    Args:
        agent: Trained agent
        env: Environment
        num_episodes: Number of evaluation episodes
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
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def train_lcsac(agent, env, edmd_model, replay_buffer, max_steps, warm_up_steps,
                train_interval, train_batch_size, eval_interval=0, eval_batch_size=10,
                log_interval=0, output_dir=None, obs_normalizer=None, norm_obs=False,
                trajectory_type=None, env_config=None, state_error_dim=6):
    """
    Train LC-SAC agent on environment.

    Args:
        agent: LC-SAC agent
        env: Environment
        edmd_model: EDMD model for Lyapunov constraint
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
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming

    Returns:
        Dictionary with training results
    """
    if output_dir is None:
        output_dir = Path("results/online/lcsac/quadrotor_2d_track/standalone")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []

    # Evaluation metrics
    eval_rewards = []
    best_eval_reward = float('-inf')

    # Initialize environment
    state, info = env.reset()
    x_ref_full = info.get('x_reference')
    if x_ref_full is None:
        raise ValueError("train_lcsac requires info['x_reference'] from the env.")
    # Stabilization: x_ref_full is constant (6,); tracking: time-series (T, 6).
    is_stab = (np.asarray(x_ref_full).ndim == 1)

    episode_reward = 0
    episode_length = 0
    total_steps = 0

    print(f"Starting LC-SAC training for {max_steps} steps (warm-up: {warm_up_steps})")

    while total_steps < max_steps:
        # Select action
        if total_steps < warm_up_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        # Step environment
        next_state, reward, done, next_info = env.step(action)

        # Calculate tracking error: constant goal (stab) or time-indexed (track)
        if is_stab:
            X_error = state[:state_error_dim] - np.asarray(x_ref_full)[:state_error_dim]
        else:
            current_step = min(episode_length, x_ref_full.shape[0] - 1)
            X_error = state[:state_error_dim] - x_ref_full[current_step][:state_error_dim]

        # Store transition
        mask = 1.0 - done.astype(np.float32) if isinstance(done, np.ndarray) else (1.0 - float(done))

        # Handle time truncation (guard for envs that don't expose ctrl_step_counter)
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
        X_error_to_push = X_error.reshape(1, -1) if X_error.ndim == 1 else X_error

        # Store transition
        transition_dict = {
            'obs': obs_to_push,
            'act': act_to_push,
            'rew': np.array([reward]),
            'next_obs': next_obs_to_push,
            'mask': np.array([true_mask]),
            'X_error': X_error_to_push
        }

        replay_buffer.push(transition_dict)

        state = next_state
        info = next_info
        episode_reward += reward
        episode_length += 1
        total_steps += 1

        # Train agent
        if (total_steps > warm_up_steps and len(replay_buffer) >= train_batch_size
            and (total_steps % train_interval == 0)):
            losses = agent.update(replay_buffer, train_batch_size)
            training_losses.append({
                'step': total_steps,
                **losses
            })

        # Handle episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Log every log_interval *steps* (step-based, consistent with eval_interval)
            num_episodes = len(episode_rewards)
            if log_interval > 0 and total_steps % log_interval == 0:
                n_recent = min(50, num_episodes)
                recent_rewards = episode_rewards[-n_recent:]
                recent_lengths = episode_lengths[-n_recent:]
                n_loss = min(log_interval, len(training_losses))
                recent_losses = training_losses[-n_loss:] if n_loss > 0 else []
                avg_lyap   = np.mean([l.get('lyap_loss',   0.0) for l in recent_losses]) if recent_losses else 0.0
                avg_critic = np.mean([l.get('critic_loss', 0.0) for l in recent_losses]) if recent_losses else 0.0
                avg_actor  = np.mean([l.get('policy_loss', 0.0) for l in recent_losses]) if recent_losses else 0.0
                print(f"Ep {num_episodes}, Steps {total_steps}, "
                      f"Reward {episode_reward:.2f} (avg {np.mean(recent_rewards):.2f}), "
                      f"Lyap {avg_lyap:.4f}, Critic {avg_critic:.4f}, Actor {avg_actor:.4f}")

            # Reset environment
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            x_ref_full = info.get('x_reference')
            if x_ref_full is None:
                raise ValueError("train_lcsac requires info['x_reference'] after env.reset().")
            is_stab = (np.asarray(x_ref_full).ndim == 1)

        # Evaluation
        if eval_interval > 0 and total_steps % eval_interval == 0 and total_steps > warm_up_steps:
            eval_reward = evaluate_agent(agent, env, eval_batch_size, obs_normalizer if norm_obs else None)
            eval_rewards.append({
                'step': total_steps,
                'reward': eval_reward
            })

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)
                best_model_path = output_dir / f"best_lcsac_model{traj_suffix}.pth"
                agent.save(str(best_model_path))
                print(f"Saved best model to {best_model_path}")

            print(f"Evaluation at step {total_steps}: Avg reward = {eval_reward:.2f} "
                  f"(Best: {best_eval_reward:.2f})")
            # evaluate_agent resets env multiple times; reset here so the next training
            # step starts from a clean env state instead of a stale one.
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            x_ref_full = info.get('x_reference')
            if x_ref_full is None:
                raise ValueError("train_lcsac requires info['x_reference'] after eval reset.")
            is_stab = (np.asarray(x_ref_full).ndim == 1)
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

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

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

    # Training losses
    if training_results['training_losses']:
        losses = training_results['training_losses']
        steps = [l['step'] for l in losses]
        axes[2, 0].plot(steps, [l.get('lyap_loss', 0.0) for l in losses],
                       label='Lyapunov Loss', color='C3')
        axes[2, 0].plot(steps, [l['critic_loss'] for l in losses],
                       label='Critic Loss', color='C0')
        axes[2, 0].plot(steps, [l.get('policy_loss', 0.0) for l in losses],
                       label='Policy Loss', color='C2')
        axes[2, 0].set_title('Training Losses')
        axes[2, 0].set_xlabel('Training Steps')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
    else:
        axes[2, 0].text(0.5, 0.5, 'No training loss data',
                        ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Training Losses')

    # Alpha (temperature parameter)
    if training_results['training_losses']:
        losses = training_results['training_losses']
        steps = [l['step'] for l in losses]
        alpha_vals = [l.get('alpha', 0.0) for l in losses]
        axes[2, 1].plot(steps, alpha_vals, color='C4')
        axes[2, 1].set_title('Alpha (Temperature)')
        axes[2, 1].set_xlabel('Training Steps')
        axes[2, 1].set_ylabel('Alpha')
        axes[2, 1].grid(True)
    else:
        axes[2, 1].text(0.5, 0.5, 'No alpha data',
                        ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Alpha (Temperature)')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"training_plots{traj_suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training plots to {plot_path}")

    plt.show()


def save_training_data(training_results, output_dir, trajectory_type=None, env_config=None,
                      max_env_steps=0, warm_up_steps=0, train_interval=0, train_batch_size=0,
                      hidden_dim=0, gamma=0, tau=0, init_temperature=0, actor_lr=0, critic_lr=0):
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
        loss_array = np.array([[l['step'], l['critic_loss'], l.get('actor_loss', 0.0),
                               l.get('policy_loss', 0.0), l.get('lyap_loss', 0.0),
                               l.get('entropy_loss', 0.0), l.get('alpha', 0.0)]
                              for l in training_results['training_losses']])
        np.save(output_dir / f"training_losses{traj_suffix}.npy", loss_array)

        # Save dedicated lyap_loss file for compare_results.py
        lyap_records = [
            {"step": l["step"], "lyap_loss": l.get("lyap_loss", 0.0)}
            for l in training_results['training_losses']
            if "step" in l
        ]
        with open(output_dir / f"lyap_loss{traj_suffix}.json", "w") as f:
            json.dump(lyap_records, f)

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
            'alpha': init_temperature,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
        }
    }

    summary_path = output_dir / f"training_summary{traj_suffix}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")

    print("\nAll training data saved successfully!")


def plot_lyapunov_loss(training_results, output_dir, trajectory_type=None, env_config=None):
    """
    Plot Lyapunov loss separately.

    Args:
        training_results: Dictionary with training results
        output_dir: Output directory
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
    """
    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)

    if not training_results['training_losses']:
        print("No training losses available for plotting")
        return

    losses = training_results['training_losses']
    steps = [l['step'] for l in losses]
    lyap_loss = [l.get('lyap_loss', 0.0) for l in losses]

    # Plot all Lyapunov loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lyap_loss, label='Lyapunov Loss', color='C5')
    plt.xlabel('Time Steps')
    plt.ylabel('Lyapunov Loss')
    plt.title('Lyapunov Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Remove spikes and plot again
    lyap_loss_no_spikes = [loss for loss in lyap_loss if loss < 1e2]
    plt.figure(figsize=(10, 6))
    plt.plot(steps[:len(lyap_loss_no_spikes)], lyap_loss_no_spikes,
            label='Lyapunov Loss', color='C5')
    plt.xlabel('Time Steps')
    plt.ylabel('Lyapunov Loss')
    plt.title('Lyapunov Loss Over Time (No Spikes)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot moving average
    window_size = 50
    if len(lyap_loss_no_spikes) >= window_size:
        lyap_ma = np.convolve(lyap_loss_no_spikes, np.ones(window_size)/window_size, mode='valid')
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(steps[:len(lyap_ma)], lyap_ma, label=r'Lyapunov Loss MA', color='C2')
        plt.xlabel(r'Time Steps $\longrightarrow$')
        plt.ylabel(r'Lyapunov Loss (MA) $\longrightarrow$')
        plt.title(f'Lyapunov Loss Moving Average (window={window_size})')
        plt.legend()

        ma_plot_path = output_dir / f"lyapunov_loss_moving_average{traj_suffix}.png"
        plt.savefig(ma_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved Lyapunov loss moving average plot to {ma_plot_path}")
        plt.show()

    # Print minimum value
    min_lyap_loss = min(lyap_loss)
    print(f"Minimum Lyapunov Loss: {min_lyap_loss}")


def evaluate_trajectory(agent, env, output_dir, trajectory_type=None, env_config=None,
                       hidden_dim=128, gamma=0.99, tau=0.005, init_temperature=0.2,
                       actor_lr=0.0001, critic_lr=0.0003, entropy_lr=0.0001,
                       use_entropy_tuning=True, edmd_model=None, P_lifted=None, A=None, B=None):
    """
    Evaluate trained agent and plot trajectory.

    Args:
        agent: Trained agent (or None to load from file)
        env: Environment
        output_dir: Output directory
        trajectory_type: Trajectory type for file naming
        env_config: Environment config for file naming
        Additional parameters for agent initialization if loading from file
    """
    traj_suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)

    # Load model if agent not provided
    if agent is None:
        model_path = output_dir / f"best_lcsac_model{traj_suffix}.pth"
        if not model_path.exists():
            model_path = output_dir / f"lcsac_model_final{traj_suffix}.pth"

        if model_path.exists():
            print(f"Loading model from {model_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            eval_agent = LCSAC(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                action_range=[env.action_space.low.copy(), env.action_space.high.copy()],
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
                use_entropy_tuning=use_entropy_tuning
            )
            eval_agent.load(str(model_path))
            eval_agent.eval()
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {model_path}")
            return

    if agent is not None:
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
        action = eval_agent.select_action(state, deterministic=True)
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
            print("Warning: Episode exceeded 10000 steps, forcing termination")
            break

    print(f"\nEvaluation episode completed: {step_count} steps")
    print(f"Total trajectory length: {len(agent_states)} states")

    # Convert to numpy arrays
    agent_states = np.array(agent_states)
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
             'r--', linewidth=2, label='LC-SAC Agent', alpha=0.7)
    ax1.scatter(ref_positions[0, 0], ref_positions[0, 1],
               c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(ref_positions[-1, 0], ref_positions[-1, 1],
               c='red', s=100, marker='*', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('XZ Trajectory: Reference vs LC-SAC Agent (2D Quadrotor)')
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


def main():
    """Main execution function."""
    # Configuration
    OUTPUT_DIR = Path("results/online/lcsac/quadrotor_2d_track/standalone")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # Load environment configuration
    ENV_CONFIG_PATH = Path("./Params/Quadrotor_2D/env_track.yaml")
    assert ENV_CONFIG_PATH.exists(), f"Missing {ENV_CONFIG_PATH}"

    with open(ENV_CONFIG_PATH, "r") as f:
        env_config = yaml.safe_load(f)["task_config"]

    print("Loaded environment config from Params/Quadrotor_2D/env_track.yaml")

    # Load SAC algorithm configuration
    SAC_CONFIG_PATH = Path("./Params/Quadrotor_2D/lcsac.yaml")
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

    # Load EDMD model and LQR matrices
    EDMD_DIR = Path("results") / "edmd" / "quadrotor_2d_track"

    print("Loading EDMD model and LQR matrices...")
    EDMD_MODEL_PATH = EDMD_DIR / 'edmd_model.pkl'
    RICCATI_PATH    = EDMD_DIR / 'lqr_matrices.npz'

    edmd_model = None
    if EDMD_MODEL_PATH.exists():
        with open(EDMD_MODEL_PATH, 'rb') as f:
            edmd_model = pickle.load(f)
        print(f"Loaded EDMD model from {EDMD_MODEL_PATH}")
    else:
        print(f"Warning: {EDMD_MODEL_PATH} not found.")

    mats = np.load(RICCATI_PATH)
    print(f"Loaded LQR matrices from {RICCATI_PATH}")

    P_lifted = mats['P']
    A_cl = mats['A_cl']
    A = mats['A_lifted']
    B = mats['B_lifted']
    K = mats['K']

    lifted_dim = A_cl.shape[0]
    print(f"A_lifted shape: {A.shape}, B_lifted shape: {B.shape}")
    print(f"P_lifted shape: {P_lifted.shape}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config settings
    sac_config['use_entropy_tuning'] = sac_config.get('use_entropy_tuning', False)
    sac_config['norm_obs'] = False

    if sac_config.get('max_env_steps', 0) < 800000:
        sac_config['max_env_steps'] = 300000

    # Extract config values
    HIDDEN_DIM = sac_config.get("hidden_dim", 128)
    GAMMA = sac_config.get("gamma", 0.99)
    TAU = sac_config.get("tau", 0.005)
    INIT_TEMPERATURE = sac_config.get("init_temperature", 0.2)
    USE_ENTROPY_TUNING = sac_config.get("use_entropy_tuning", False)
    TRAIN_INTERVAL = sac_config.get("train_interval", 10)
    TRAIN_BATCH_SIZE = sac_config.get("train_batch_size", 256)
    ACTOR_LR = sac_config.get("actor_lr", 3e-4)
    CRITIC_LR = sac_config.get("critic_lr", 1e-4)
    ENTROPY_LR = sac_config.get("entropy_lr", 1e-4)
    MAX_ENV_STEPS = sac_config.get("max_env_steps", 1000000)
    WARM_UP_STEPS = sac_config.get("warm_up_steps", 1000)
    MAX_BUFFER_SIZE = sac_config.get("max_buffer_size", 1000000)
    EVAL_BATCH_SIZE = sac_config.get("eval_batch_size", 10)
    LOG_INTERVAL = sac_config.get("log_interval", 100)
    EVAL_INTERVAL = sac_config.get("eval_interval", 4000)
    LYAP_RAMP_STEPS = sac_config.get("lyap_ramp_steps", 50000)

    print(f"LC-SAC Config: hidden_dim={HIDDEN_DIM}, batch_size={TRAIN_BATCH_SIZE}, max_steps={MAX_ENV_STEPS}")
    if USE_ENTROPY_TUNING:
        print("✓ Using automatic entropy tuning")
    print(f"Actor LR: {ACTOR_LR}, Critic LR: {CRITIC_LR}")

    # Initialize environment and agent
    env = make_quadrotor_2d_env(gui=False, trajectory_type=TRAJECTORY_TYPE, env_config=env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [env.action_space.low.copy(), env.action_space.high.copy()]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action range: [{action_range[0]}, {action_range[1]}]")

    QUADTYPE = "quadrotor_2D"

    # Physical action bounds for EDMD B-matrix rescaling (needed when normalized_rl_action_space=True)
    edmd_act_low, edmd_act_high = None, None
    if env_config.get("normalized_rl_action_space", False):
        for c in env_config.get("constraints", []):
            if c.get("constrained_variable") == "input":
                lb, ub = c.get("lower_bounds"), c.get("upper_bounds")
                if lb and ub:
                    edmd_act_low = np.array(lb[:action_dim], dtype=np.float32)
                    edmd_act_high = np.array(ub[:action_dim], dtype=np.float32)
                    break

    # Initialize LC-SAC agent
    agent = LCSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        hidden_dim=HIDDEN_DIM,
        device=device,
        edmd_model=edmd_model,
        P_lifted=P_lifted,
        A=A,
        B=B,
        gamma=GAMMA,
        init_temperature=INIT_TEMPERATURE,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        entropy_lr=ENTROPY_LR,
        use_entropy_tuning=USE_ENTROPY_TUNING,
        quadtype=QUADTYPE,
        lyap_ramp_steps=LYAP_RAMP_STEPS,
        edmd_action_low=edmd_act_low,
        edmd_action_high=edmd_act_high,
    )

    # Initialize replay buffer
    replay_buffer = SACBuffer(
        obs_space=env.observation_space,
        act_space=env.action_space,
        max_size=MAX_BUFFER_SIZE,
        batch_size=TRAIN_BATCH_SIZE
    )
    print(f"✓ Initialized SACBuffer with max_size={MAX_BUFFER_SIZE}")

    print("Environment and agent initialized successfully!")

    # Training
    print("Starting LC-SAC training...")
    print(f"Config: max_steps={MAX_ENV_STEPS}, warm_up={WARM_UP_STEPS}, "
          f"train_interval={TRAIN_INTERVAL}, eval_interval={EVAL_INTERVAL}")

    training_results = train_lcsac(
        agent=agent,
        env=env,
        edmd_model=edmd_model,
        replay_buffer=replay_buffer,
        max_steps=MAX_ENV_STEPS,
        warm_up_steps=WARM_UP_STEPS,
        train_interval=TRAIN_INTERVAL,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_interval=EVAL_INTERVAL if EVAL_INTERVAL > 0 else 0,
        eval_batch_size=EVAL_BATCH_SIZE,
        log_interval=LOG_INTERVAL if LOG_INTERVAL > 0 else 1,
        output_dir=OUTPUT_DIR,
        obs_normalizer=None,
        norm_obs=False,
        trajectory_type=TRAJECTORY_TYPE,
        env_config=env_config
    )

    print("\nTraining completed!")
    print(f"Total episodes: {len(training_results['episode_rewards'])}")
    print(f"Best evaluation reward: {training_results['best_eval_reward']:.2f}")
    print(f"Average final 10 episode rewards: {np.mean(training_results['episode_rewards'][-10:]):.2f}")

    # Plot and save training results
    plot_training_results(training_results, OUTPUT_DIR, TRAJECTORY_TYPE, env_config)
    save_training_data(training_results, OUTPUT_DIR, TRAJECTORY_TYPE, env_config,
                      MAX_ENV_STEPS, WARM_UP_STEPS, TRAIN_INTERVAL, TRAIN_BATCH_SIZE,
                      HIDDEN_DIM, GAMMA, TAU, INIT_TEMPERATURE, ACTOR_LR, CRITIC_LR)

    # Plot Lyapunov loss
    plot_lyapunov_loss(training_results, OUTPUT_DIR, TRAJECTORY_TYPE, env_config)

    # Save final model
    traj_suffix = get_traj_suffix(trajectory_type=TRAJECTORY_TYPE, env_config=env_config)
    final_model_path = OUTPUT_DIR / f"lcsac_model_final{traj_suffix}.pth"
    agent.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    # Evaluate trajectory
    eval_env = make_quadrotor_2d_env(
        gui=False,
        override_config={'randomized_init': False},
        trajectory_type=TRAJECTORY_TYPE,
        env_config=env_config
    )

    evaluate_trajectory(
        agent=agent,
        env=eval_env,
        output_dir=OUTPUT_DIR,
        trajectory_type=TRAJECTORY_TYPE,
        env_config=env_config,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        tau=TAU,
        init_temperature=INIT_TEMPERATURE,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        entropy_lr=ENTROPY_LR,
        use_entropy_tuning=USE_ENTROPY_TUNING,
        edmd_model=edmd_model,
        P_lifted=P_lifted,
        A=A,
        B=B
    )

    # Close environments
    env.close()
    eval_env.close()
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
