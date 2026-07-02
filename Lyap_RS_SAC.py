"""
Reward-Shaping Lyapunov SAC (lyap_rs_sac).

Standard SAC (SACAgent) with Lyapunov-based potential shaping at collection time:

    r_shaped(t) = r(t) + w * (V(z_t) - γ * V(z_{t+1}))

where V(z) = z^T P z, z = Φ(x_error) is the EDMD-lifted state error, w is
lyap_rs_weight, and the formula is the potential-based reward shaping of Ng et al.
(1999) with potential Φ(s) = -w * V(z(s)).

The shaped reward is stored in the replay buffer so the critic learns the shaped
value function directly.  Episode rewards in the returned results are the
*unshifted* environmental rewards for fair comparison with SAC/LC-SAC.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from safe_control_gym.controllers.sac.sac_utils import SACAgent

from SAC import evaluate_agent, get_traj_suffix


def _compute_V_np(x_error: np.ndarray, edmd_model, P: np.ndarray) -> float:
    """Compute V(z) = z^T P z for a single state-error vector (raw, no bias correction)."""
    z = edmd_model.observables.transform(x_error.reshape(1, -1))
    return float(z @ P @ z.T)


def _compute_V_bias(edmd_model, P: np.ndarray, state_error_dim: int) -> float:
    """Compute V(z(0)) — the RBF offset at zero error.

    RBF lifting φ_i(0) = r(0, c_i) ≠ 0, so V(0) ≠ 0.  Subtracting this
    bias makes V_adj(0) = 0, satisfying the Ng et al. (1999) requirement
    Φ(goal) = 0 for potential-based reward shaping.
    """
    return _compute_V_np(np.zeros(state_error_dim), edmd_model, P)


def calibrate_lyap_rs_weight(
    env,
    edmd_model,
    P_np: np.ndarray,
    gamma: float = 0.99,
    n_steps: int = 500,
    state_error_dim: int = 6,
) -> float:
    """Estimate a suitable lyap_rs_weight from random rollouts.

    Computes the ratio  E[|r|] / E[|V(z_t) - γ·V(z_{t+1})|]  over n_steps
    random-action transitions so that the Lyapunov shaping term is on the same
    scale as the environment reward.  Resets env state before returning.

    Args:
        env:        The training environment (will be reset before and after).
        edmd_model: Fitted EDMD/PyKoopman model with .observables.transform().
        P_np:       Riccati matrix (N_lift × N_lift) for V(z) = z^T P z.
        gamma:      Discount used in potential shaping (must match training gamma).
        n_steps:    Number of random transitions to collect.

    Returns:
        Suggested lyap_rs_weight (float > 0).
    """
    V_bias = _compute_V_bias(edmd_model, P_np, state_error_dim)

    state, info = env.reset()
    x_ref_full = info.get('x_reference')
    is_stab = x_ref_full is not None and np.asarray(x_ref_full).ndim == 1

    rewards_abs: list[float] = []
    delta_V_abs: list[float] = []
    V_vals:      list[float] = []
    episode_length = 0

    for _ in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if x_ref_full is not None:
            ref = np.asarray(x_ref_full)
            if is_stab:
                x_err      = state[:state_error_dim]      - ref[:state_error_dim]
                x_err_next = next_state[:state_error_dim] - ref[:state_error_dim]
            else:
                idx      = min(episode_length,     ref.shape[0] - 1)
                idx_next = min(episode_length + 1, ref.shape[0] - 1)
                x_err      = state[:state_error_dim]      - ref[idx][:state_error_dim]
                x_err_next = next_state[:state_error_dim] - ref[idx_next][:state_error_dim]

            V_curr = _compute_V_np(x_err,      edmd_model, P_np) - V_bias
            V_next = _compute_V_np(x_err_next, edmd_model, P_np) - V_bias
            delta_V = V_curr - gamma * V_next

            rewards_abs.append(abs(reward))
            delta_V_abs.append(abs(delta_V))
            V_vals.append(V_curr)

        episode_length += 1
        if done:
            state, info = env.reset()
            x_ref_full = info.get('x_reference')
            is_stab = x_ref_full is not None and np.asarray(x_ref_full).ndim == 1
            episode_length = 0
        else:
            state = next_state

    # Reset env so training starts from a clean state
    env.reset()

    if not rewards_abs:
        print("[calibrate_lyap_rs_weight] No x_reference in env — returning weight=1.0")
        return 1.0

    rewards_abs = np.array(rewards_abs)
    delta_V_abs = np.array(delta_V_abs)
    V_vals      = np.array(V_vals)

    mean_r  = float(np.mean(rewards_abs))
    mean_dV = float(np.mean(delta_V_abs))
    mean_V  = float(np.mean(V_vals))
    p10_V   = float(np.percentile(V_vals, 10))
    p90_V   = float(np.percentile(V_vals, 90))

    if mean_dV < 1e-12:
        print("[calibrate_lyap_rs_weight] |ΔV| ≈ 0; EDMD may be constant. Returning weight=1.0")
        return 1.0

    # Full-distribution estimate (dominated by chaotic random-action states)
    w_global = mean_r / mean_dV

    # Near-tracking estimate: use only the lowest-V quartile, which is more
    # representative of the trained policy's operating regime (small tracking error).
    q25 = float(np.percentile(V_vals, 25))
    low_mask = V_vals <= q25
    if low_mask.sum() >= 10:
        w_near = float(np.mean(rewards_abs[low_mask])) / float(np.mean(delta_V_abs[low_mask]))
    else:
        w_near = w_global

    # Use the near-tracking estimate; it better matches what the trained policy sees.
    suggested = w_near

    print(
        f"[Lyap-RS calibration over {n_steps} random steps]\n"
        f"  V(zero error) bias = {V_bias:.5f}  (subtracted from all V; V_adj(0)=0)\n"
        f"  E[V_adj]          = {mean_V:.4f}  (p10={p10_V:.4f}, p90={p90_V:.4f})\n"
        f"  Global  E[|r|]/E[|ΔV|] = {w_global:.5f}  (random-action regime)\n"
        f"  Near-tracking E[|r|]/E[|ΔV|] (V ≤ p25={q25:.2f}) = {w_near:.5f}\n"
        f"  → using lyap_rs_weight = {suggested:.4f}  (near-tracking estimate)"
    )
    return suggested


def train_lyap_rs_sac(
    agent: SACAgent,
    env,
    edmd_model,
    P_np: np.ndarray,
    replay_buffer,
    max_steps: int,
    warm_up_steps: int,
    train_interval: int,
    train_batch_size: int,
    lyap_rs_weight: float = 1.0,
    gamma: float = 0.99,
    eval_interval: int = 0,
    eval_batch_size: int = 10,
    log_interval: int = 0,
    output_dir: Path | None = None,
    device=None,
    trajectory_type: str | None = None,
    env_config: dict | None = None,
    sac_config: dict | None = None,
    state_error_dim: int = 6,
) -> dict:
    """Train Lyapunov-reward-shaping SAC.

    Identical to train_sac except each stored reward is shaped with V-based
    potential before being pushed to the replay buffer.
    """
    if output_dir is None:
        output_dir = Path("results/online/lyap_rs_sac/quadrotor_2d_track/standalone")

    # Precompute RBF offset so V_adj(zero_error) = 0, satisfying Φ(goal)=0
    V_bias = _compute_V_bias(edmd_model, P_np, state_error_dim)
    print(f"  V_bias (RBF offset at zero error) = {V_bias:.5f}")

    episode_rewards, episode_lengths, training_losses = [], [], []
    eval_rewards = []
    # lyap_entries: per-step Lyapunov metrics saved to lyap_loss{suffix}.json.
    # lyap_loss  = relu(V_next - V_curr) — same violation metric as LC-SAC (decay_rate=0).
    # lyap_shaping = V_curr - gamma*V_next — the actual reward-shaping signal.
    lyap_entries: list[dict] = []
    best_eval_reward = float('-inf')

    state, info = env.reset()
    x_ref_full = info.get('x_reference')
    is_stab = x_ref_full is not None and np.asarray(x_ref_full).ndim == 1

    episode_reward = 0
    episode_length = 0
    total_steps = 0

    print(f"Starting Lyap-RS-SAC for {max_steps} steps  (lyap_rs_weight={lyap_rs_weight:.4f})")
    agent.train()

    while total_steps < max_steps:
        # ------ action selection ------
        if total_steps < warm_up_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.ac.act(
                    torch.FloatTensor(state).unsqueeze(0).to(device), deterministic=False)
                if isinstance(action, np.ndarray) and action.ndim > 0:
                    action = action[0]

        next_state, reward, done, info = env.step(action)

        # ------ Lyapunov potential shaping ------
        shaping = 0.0
        if x_ref_full is not None:
            ref = np.asarray(x_ref_full)
            if is_stab:
                x_err      = state[:state_error_dim]      - ref[:state_error_dim]
                x_err_next = next_state[:state_error_dim] - ref[:state_error_dim]
            else:
                idx      = min(episode_length,     ref.shape[0] - 1)
                idx_next = min(episode_length + 1, ref.shape[0] - 1)
                x_err      = state[:state_error_dim]      - ref[idx][:state_error_dim]
                x_err_next = next_state[:state_error_dim] - ref[idx_next][:state_error_dim]
            V_curr = _compute_V_np(x_err,      edmd_model, P_np) - V_bias
            V_next = _compute_V_np(x_err_next, edmd_model, P_np) - V_bias
            # Potential-based shaping: F(s,a,s') = γΦ(s') - Φ(s), Φ = -w*V_adj
            shaping = lyap_rs_weight * (V_curr - gamma * V_next)
            lyap_entries.append({
                "step": total_steps,
                # lyap_loss: relu(V_next - V_curr) — same violation metric as LC-SAC (decay_rate=0)
                "lyap_loss": float(max(0.0, V_next - V_curr)),
                "lyap_shaping": float(shaping),
                "V_curr": float(V_curr),
            })

        reward_shaped = reward + shaping

        # ------ store transition ------
        mask = 1.0 - float(done)
        is_timeout = (done
                      and hasattr(env, 'ctrl_step_counter')
                      and hasattr(env, 'CTRL_STEPS')
                      and env.ctrl_step_counter >= env.CTRL_STEPS)
        true_mask = 1.0 if is_timeout else mask

        replay_buffer.push({
            'obs':      state.reshape(1, -1),
            'act':      action.reshape(1, -1),
            'rew':      np.array([reward_shaped]),
            'next_obs': next_state.reshape(1, -1),
            'mask':     np.array([true_mask]),
        })

        state = next_state
        episode_reward += reward    # track unshifted reward for apples-to-apples comparison
        episode_length += 1
        total_steps += 1

        # ------ SAC gradient update ------
        if (total_steps > warm_up_steps
                and len(replay_buffer) >= train_batch_size
                and total_steps % train_interval == 0):
            batch = replay_buffer.sample(batch_size=train_batch_size, device=device)
            losses = agent.update(batch)
            training_losses.append({
                'step':         total_steps,
                'critic_loss':  losses.get('critic_loss', 0.0),
                'actor_loss':   losses.get('policy_loss', 0.0),
                'entropy_loss': losses.get('entropy_loss', 0.0),
                'alpha':        agent.alpha.item(),
            })

        # ------ episode end ------
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if log_interval > 0 and total_steps % log_interval == 0:
                recent = episode_rewards[-min(50, len(episode_rewards)):]
                print(f"Ep {len(episode_rewards)}, Steps {total_steps}, "
                      f"Reward {episode_reward:.2f} (avg {np.mean(recent):.2f})")
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            x_ref_full = info.get('x_reference')
            if x_ref_full is not None:
                is_stab = np.asarray(x_ref_full).ndim == 1

        # ------ evaluation ------
        if eval_interval > 0 and total_steps % eval_interval == 0 and total_steps > warm_up_steps:
            eval_reward = evaluate_agent(agent, env, eval_batch_size, device=device)
            eval_rewards.append({'step': total_steps, 'reward': eval_reward})
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                if sac_config and sac_config.get('eval_save_best', False):
                    suffix = get_traj_suffix(trajectory_type=trajectory_type, env_config=env_config)
                    best_path = output_dir / f"best_lyap_rs_sac_model{suffix}.pth"
                    torch.save(agent.state_dict(), str(best_path))
                    print(f"Saved best model → {best_path}")
            print(f"Eval step {total_steps}: {eval_reward:.2f} (best {best_eval_reward:.2f})")
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            x_ref_full = info.get('x_reference')
            if x_ref_full is not None:
                is_stab = np.asarray(x_ref_full).ndim == 1
            agent.train()

    return {
        'episode_rewards':  episode_rewards,
        'episode_lengths':  episode_lengths,
        'training_losses':  training_losses,
        'eval_rewards':     eval_rewards,
        'best_eval_reward': best_eval_reward,
        'lyap_entries':     lyap_entries,
    }
