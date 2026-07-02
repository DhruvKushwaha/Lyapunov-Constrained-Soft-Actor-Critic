"""
LCSACMeanAgent: Barrier-Lyapunov Actor-Critic with Lyapunov (CLF) constraint only.

Based on Zhao et al., "Stable and Safe Reinforcement Learning via a Barrier-Lyapunov
Actor-Critic Approach" (CDC 2023), with the CBF/barrier term removed so only the CLF
stability constraint remains. This makes it a direct ablation of LC-SAC.

Inherits network architecture and training infrastructure from LCSAC (LC_SAC.py).
Key algorithmic differences from LCSAC:
  - Mean violation in actor loss (not CVaR 0.9)
  - No ramp-in schedule; lambda starts at 0 and adapts immediately
  - Bidirectional lambda update (can decrease when constraint satisfied)
  - Larger CLF decay margin: alpha_V = 0.01 (vs DECAY_RATE = 1e-3 in LC-SAC)
  - Larger lambda_max = 1.0 (vs 0.1 in LC-SAC)

Compatible with train_lcsac() in LC_SAC_Train.py (same buffer interface).
"""
import numpy as np
import torch
import torch.nn.functional as F

from LC_SAC import LCSAC


class LCSACMeanAgent(LCSAC):
    """LC-SAC with mean (not CVaR) violation aggregation.

    Args:
        alpha_V: CLF exponential decay rate for decrease margin.
                 Constraint: V(z') <= (1 - alpha_V) * V(z).
        lam_lr:  Learning rate for adaptive Lagrange multiplier.
        lam_max: Upper bound on Lagrange multiplier.
    """

    def __init__(self, *args, alpha_V: float = 0.01, lam_lr: float = 1e-4,
                 lam_max: float = 1.0, **kwargs):
        kwargs.setdefault("lyap_ramp_steps", 0)
        super().__init__(*args, **kwargs)
        self.alpha_V = float(alpha_V)
        self.lam_lr = float(lam_lr)
        # Replace parent's Parameter lam with a plain scalar (bidirectional update)
        self.lam = 0.0
        self.lam_max = float(lam_max)

    def update(self, replay_buffer, batch_size):
        self.train()
        batch = replay_buffer.sample(batch_size, self.device)
        state     = batch['obs']
        action    = batch['act']
        reward    = batch['rew']
        next_state = batch['next_obs']
        mask      = batch['mask']
        X_error   = batch['X_error']

        # ---------- Critic update (identical to LCSAC) ----------
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        with torch.no_grad():
            a_next, log_pi_next = self.actor(next_state, deterministic=False, with_logprob=True)
            tq1 = self.critic_1_target(next_state, a_next)
            tq2 = self.critic_2_target(next_state, a_next)
            min_tq = torch.min(tq1, tq2) - self.alpha * log_pi_next
            target_q = reward + self.gamma * mask * min_tq

        q1_loss = F.huber_loss(current_q1, target_q)
        q2_loss = F.huber_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), max_norm=10.0)
        self.critic_optimizer.step()

        # ---------- Actor update with LC-SAC-Mean Lyapunov constraint ----------
        action_new, log_prob = self.actor(state, deterministic=False, with_logprob=True)

        # Freeze critics during actor update
        for p in list(self.critic_1.parameters()) + list(self.critic_2.parameters()):
            p.requires_grad_(False)
        q_new = torch.min(self.critic_1(state, action_new), self.critic_2(state, action_new))
        for p in list(self.critic_1.parameters()) + list(self.critic_2.parameters()):
            p.requires_grad_(True)

        policy_loss = ((self.alpha.detach() * log_prob) - q_new).mean()

        # Lyapunov CLF decrease: V(z') <= (1 - alpha_V) * V(z)
        # Equivalently: relu(V_next - V_curr + alpha_V * V_curr) = violation
        z      = self.lift_state(X_error)
        z_next = z @ self.A.T + self._to_edmd_action(action_new) @ self.B.T
        V_curr = self.Lyapunov_fn(z) - self.V_bias
        V_next = self.Lyapunov_fn(z_next) - self.V_bias

        # Raw violation (signed): negative when constraint is satisfied with margin.
        # V(z') <= (1 - alpha_V)*V(z)  ⟺  V_next - V_curr + alpha_V*V_curr <= 0
        raw_violation = (V_next - V_curr + self.alpha_V * V_curr) * mask

        # Actor loss uses relu (non-negative) so gradients flow only on actual violations.
        lyap_loss = F.relu(raw_violation).mean()

        # Linear Lagrangian actor loss (no ramp-in)
        total_actor_loss = policy_loss + self.lam * lyap_loss
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # Bidirectional lambda update: uses signed violation so lam decreases when
        # the constraint is satisfied (raw_violation < 0), per Zhao et al. CDC 2023.
        signed_mean = raw_violation.mean().item()
        self.lam = float(
            np.clip(self.lam + self.lam_lr * signed_mean, 0.0, self.lam_max))

        # ---------- Entropy coefficient ----------
        entropy_loss = torch.zeros(1)
        if self.use_entropy_tuning:
            entropy_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()

        # ---------- Soft target update ----------
        self._soft_update(self.critic_1_target, self.critic_1)
        self._soft_update(self.critic_2_target, self.critic_2)

        return {
            'critic_loss':  critic_loss.item(),
            'actor_loss':   total_actor_loss.item(),
            'policy_loss':  policy_loss.item(),
            'lyap_loss':    lyap_loss.item(),
            'lam':          self.lam,
            'lyap_ramp':    1.0,  # always active; keeps compatibility with train_lcsac logging
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else 0.0,
            'alpha':        self.alpha.detach().item(),
        }

    # ------------------------------------------------------------------
    # save / load — lam is a plain float, not a Parameter
    # ------------------------------------------------------------------

    def save(self, filepath):
        torch.save({
            'actor':            self.actor.state_dict(),
            'critic_1':         self.critic_1.state_dict(),
            'critic_2':         self.critic_2.state_dict(),
            'critic_1_target':  self.critic_1_target.state_dict(),
            'critic_2_target':  self.critic_2_target.state_dict(),
            'actor_optimizer':  self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha':        self.log_alpha,
            'lam':              self.lam,
        }, filepath)

    def load(self, filepath):
        ck = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(ck['actor'])
        self.critic_1.load_state_dict(ck['critic_1'])
        self.critic_2.load_state_dict(ck['critic_2'])
        self.critic_1_target.load_state_dict(ck.get('critic_1_target', ck['critic_1']))
        self.critic_2_target.load_state_dict(ck.get('critic_2_target', ck['critic_2']))
        self.actor_optimizer.load_state_dict(ck['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ck['critic_optimizer'])
        if 'log_alpha' in ck and ck['log_alpha'] is not None:
            self.log_alpha.data.copy_(ck['log_alpha'].to(self.device).data)
        if 'lam' in ck:
            self.lam = float(ck['lam'])
