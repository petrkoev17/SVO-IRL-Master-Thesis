"""
IQ-Learn implementation based on "IQ-Learn: Inverse soft-Q Learning for Imitation" (Garg et al., 2021)

Supports optional SVO regularization with three modes:

    'bellman'    – (Original) Shift the Bellman target by λ·R_SVO.
                   Pros: simple.  Cons: SVO and γV(s') compete on different
                   scales; the ratio drifts during training.

    'reward_reg' – (Recommended) Add a separate MSE loss that pushes the
                   *recovered* reward  r̂(s,a) = Q(s,a) − γV(s')  toward
                   λ·R_SVO.  The IQ-Learn objective stays structurally
                   intact; the SVO term acts as a soft prior on reward shape.

    'reweight'   – Use R_SVO to importance-weight expert transitions during
                   sampling.  High-SVO-alignment transitions are sampled more
                   often; adversarial ones are down-weighted.  The loss itself
                   is untouched.

When use_svo=False (default), behaviour is identical to standard IQ-Learn
regardless of svo_mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import gymnasium as gym


# ======================================================================
# Replay Buffer
# ======================================================================

class ReplayBuffer:
    """
    Replay Buffer for expert demonstrations and learner experience.

    Each transition stores:
        (state, action, reward, next_state, done, r_self, r_global)

    r_self and r_global are the raw SVO reward components. They are always
    stored (defaulting to 0.0 when unavailable) so the buffer format is
    uniform regardless of whether SVO regularization is active.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done,
            r_self: float = 0.0, r_global: float = 0.0):
        self.buffer.append((state, action, reward, next_state, done, r_self, r_global))

    def add_trajectory(self, trajectory: List[Tuple]):
        """
        Add entire trajectory to the buffer.

        Supported tuple formats:
            5-element: (s, a, r, s', done)                              — minimal
            6-element: (s, a, r, s', done, crashed)                     — legacy
            8-element: (s, a, r, s', done, crashed, r_self, r_global)   — current
        """
        for transition in trajectory:
            n = len(transition)
            if n >= 8:
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                    float(transition[6]), float(transition[7]),
                )
            elif n == 6:
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                )
            else:
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                )

    # ------------------------------------------------------------------
    # Uniform sampling (default)
    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return self._gather(indices)

    # ------------------------------------------------------------------
    # SVO-weighted sampling  (used by svo_mode='reweight')
    # ------------------------------------------------------------------
    def sample_weighted(self, batch_size: int, weights: np.ndarray):
        """
        Sample with pre-computed per-transition importance weights.

        Args:
            weights: 1-D array of length len(self.buffer). Need NOT sum to 1;
                     will be normalised internally.
        """
        probs = weights / weights.sum()
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=True, p=probs)
        return self._gather(indices)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _gather(self, indices):
        batch = [self.buffer[idx] for idx in indices]

        states      = torch.FloatTensor(np.array([x[0].flatten() for x in batch]))
        actions     = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards     = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3].flatten() for x in batch]))
        dones       = torch.FloatTensor(np.array([x[4] for x in batch]))
        r_selfs     = torch.FloatTensor(np.array([x[5] for x in batch]))
        r_globals   = torch.FloatTensor(np.array([x[6] for x in batch]))

        return states, actions, rewards, next_states, dones, r_selfs, r_globals

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# Q-Network
# ======================================================================

class DQNNetwork(nn.Module):
    """Q-Network"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# ======================================================================
# IQ-Learn Trainer
# ======================================================================

class IQLearnTrainer:
    """
    IQ-Learn Trainer for learning from expert demonstrations.

    SVO modes (only active when use_svo=True):
        'bellman'     – shift Bellman target by λ·R_SVO  (original)
        'reward_reg'  – separate MSE loss on recovered reward vs λ·R_SVO
        'reweight'    – importance-weight expert sampling by R_SVO
    """

    VALID_SVO_MODES = ('bellman', 'reward_reg', 'reweight', 'reward_reg_reweight')

    def __init__(self,
                 env: gym.Env,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # IQ-Learn core
                 method: str = 'value',
                 loss_type: str = 'v0',
                 regularize_weight: float = 1.0,
                 temperature: float = 1.0,
                 learner_buffer_size: int = 5_000,
                 # Stabilization
                 gradient_penalty_weight: float = 0.1,
                 replay_ratio: float = 0.5,
                 # SVO regularization
                 use_svo: bool = False,
                 svo_mode: str = 'reward_reg',
                 svo_alpha: float = 0.0,
                 svo_lambda: float = 1.0,
                 normalize_svo: bool = False,
                 # Reweight parameters
                 svo_reweight_temp: float = 1.0,
    ):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.method = method
        self.loss_type = loss_type
        self.regularize_weight = regularize_weight
        self.temperature = temperature
        self.gradient_penalty_weight = gradient_penalty_weight
        self.replay_ratio = replay_ratio

        # --- SVO config ---
        self.use_svo = use_svo
        self.svo_mode = svo_mode
        self.svo_alpha = svo_alpha
        self.svo_lambda = svo_lambda
        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)
        self.normalize_svo = normalize_svo
        self.svo_reweight_temp = svo_reweight_temp

        # Precomputed importance weights for 'reweight' mode (set after loading demos)
        self._expert_svo_weights: Optional[np.ndarray] = None

        if self.use_svo:
            if self.svo_mode not in self.VALID_SVO_MODES:
                raise ValueError(
                    f"Invalid svo_mode='{svo_mode}'. "
                    f"Must be one of {self.VALID_SVO_MODES}"
                )
            print(f"[SVO-IQ] SVO regularization ENABLED")
            print(f"[SVO-IQ]   mode       = {svo_mode}")
            print(f"[SVO-IQ]   α_target   = {np.degrees(svo_alpha):.1f}°  ({svo_alpha:.4f} rad)")
            print(f"[SVO-IQ]   λ          = {svo_lambda}")
            print(f"[SVO-IQ]   normalize  = {normalize_svo}")
            if svo_mode == 'reweight':
                print(f"[SVO-IQ]   reweight τ = {svo_reweight_temp}")
        else:
            print(f"[IQ-Learn] Standard IQ-Learn (no SVO regularization)")

        # Q-networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q_target  = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Buffers
        self.expert_buffer  = ReplayBuffer()
        self.learner_buffer = ReplayBuffer(capacity=learner_buffer_size)

        # Logging
        self.losses           = []
        self.expert_q_values  = []
        self.learner_q_values = []
        self.abs_q_values     = []
        self.svo_ratios       = []
        self.abs_svo_shifts   = []

    # ==================================================================
    # Expert loading
    # ==================================================================

    def load_expert_demonstrations(self, demonstrations: List[List[Tuple]]):
        """Load expert demonstrations into expert buffer."""
        print(f"Loading {len(demonstrations)} expert demonstrations")

        has_svo = False
        if demonstrations and demonstrations[0]:
            has_svo = len(demonstrations[0][0]) >= 8

        if self.use_svo and not has_svo:
            print(
                "[SVO-IQ] WARNING: SVO regularization is enabled but "
                "demonstrations do not contain r_self / r_global. "
                "SVO reward shaping will use zeros (no effect)."
            )

        for traj in demonstrations:
            self.expert_buffer.add_trajectory(traj)
        print(f"Expert buffer size: {len(self.expert_buffer)}")

        # Pre-compute importance weights for reweight modes
        if self.use_svo and self.svo_mode in ('reweight', 'reward_reg_reweight'):
            self._compute_expert_weights()

    def _compute_expert_weights(self):
        """
        Compute per-transition importance weights from R_SVO.

        w_i = softmax( R_SVO_i / τ )

        Higher-alignment transitions get higher weight.  Temperature τ
        controls how peaked the distribution is:
            τ → 0  :  hard selection (only the best transitions)
            τ → ∞  :  uniform (no effect)
        """
        n = len(self.expert_buffer)
        svo_vals = np.empty(n, dtype=np.float32)

        for i, (_, _, _, _, _, r_self, r_global) in enumerate(self.expert_buffer.buffer):
            svo_vals[i] = self.cos_alpha * r_self + self.sin_alpha * r_global

        # Softmax with temperature
        logits = svo_vals / (self.svo_reweight_temp + 1e-8)
        logits -= logits.max()  # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum()

        self._expert_svo_weights = weights

        print(f"[SVO-IQ reweight] Weight stats: "
              f"min={weights.min():.6f}  max={weights.max():.6f}  "
              f"effective_N={1.0 / (weights**2).sum():.0f} / {n}")

    # ==================================================================
    # SVO helpers
    # ==================================================================

    def _compute_svo_reward(self, r_selfs: torch.Tensor,
                            r_globals: torch.Tensor) -> torch.Tensor:
        """R_SVO = cos(α) · r_self + sin(α) · r_global   →  [batch, 1]"""
        r_svo = self.cos_alpha * r_selfs + self.sin_alpha * r_globals
        return r_svo.unsqueeze(1)

    # ==================================================================
    # Value functions
    # ==================================================================

    def soft_value_function(self, states: torch.Tensor) -> torch.Tensor:
        q_values = self.q_network(states)
        return self.temperature * torch.logsumexp(
            q_values / self.temperature, dim=1, keepdim=True)

    def soft_value_function_target(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.q_target(states)
            return self.temperature * torch.logsumexp(
                q_values / self.temperature, dim=1, keepdim=True)

    # ==================================================================
    # Loss computation
    # ==================================================================

    def compute_iq_loss(self,
                        expert_states, expert_actions,
                        expert_next_states, expert_dones,
                        expert_r_selfs, expert_r_globals,
                        learner_states, learner_actions,
                        learner_next_states, learner_dones,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute IQ-Learn loss, optionally with SVO regularization.

        The SVO integration point depends on self.svo_mode:
            'bellman'    – target = λ·R_SVO + γ(1-d)V(s')
            'reward_reg' – target = γ(1-d)V(s'),  plus separate L_svo
            'reweight'   – target = γ(1-d)V(s')  (sampling already biased)
        """

        # ---- Expert Q and V ----
        expert_q      = self.q_network(expert_states).gather(
                            1, expert_actions.unsqueeze(1))
        expert_next_v = self.soft_value_function_target(expert_next_states)
        gamma_v_next  = self.gamma * (1 - expert_dones.unsqueeze(1)) * expert_next_v

        # ---- Learner Q and V ----
        if len(learner_states) > 0:
            learner_q      = self.q_network(learner_states).gather(
                                1, learner_actions.unsqueeze(1))
            learner_next_v = self.soft_value_function_target(learner_next_states)
        else:
            learner_q      = torch.zeros_like(expert_q)
            learner_next_v = torch.zeros_like(expert_next_v)

        # ---- Compute SVO terms (needed for bellman, reward_reg, reward_reg_reweight) ----
        svo_shift = 0.0          # scalar zero — broadcasts cleanly
        r_svo_raw = None         # for logging

        if self.use_svo and self.svo_mode in ('bellman', 'reward_reg', 'reward_reg_reweight'):
            r_svo_raw = self._compute_svo_reward(
                expert_r_selfs, expert_r_globals).to(self.device)

            if self.normalize_svo:
                mu  = r_svo_raw.mean()
                std = r_svo_raw.std() + 1e-8
                r_svo = (r_svo_raw - mu) / std
            else:
                r_svo = r_svo_raw

            svo_shift = self.svo_lambda * r_svo

        # ---- IQ-Learn loss (v0 / v1) ----
        if self.loss_type == 'v0':

            # Bellman mode: shift the target
            if self.use_svo and self.svo_mode == 'bellman':
                expert_target = svo_shift + gamma_v_next
            else:
                expert_target = gamma_v_next

            expert_bellman_residual = expert_q - expert_target

            if len(learner_states) > 0:
                learner_target = self.gamma * (1 - learner_dones.unsqueeze(1)) * learner_next_v
                learner_bellman_residual = learner_q - learner_target

                expert_loss  = (expert_bellman_residual ** 2).mean()
                learner_loss = 0.5 * F.relu(learner_bellman_residual).mean()
                loss = expert_loss + self.regularize_weight * learner_loss
            else:
                loss = (expert_bellman_residual ** 2).mean()

            grad_penalty = (expert_q ** 2).mean()
            loss = loss + self.gradient_penalty_weight * grad_penalty

        elif self.loss_type == 'v1':

            if self.use_svo and self.svo_mode == 'bellman':
                expert_loss = -(expert_q - svo_shift).mean()
            else:
                expert_loss = -expert_q.mean()

            if len(learner_states) > 0:
                learner_loss = learner_q.mean()
                loss = expert_loss + self.regularize_weight * learner_loss
            else:
                loss = expert_loss

            grad_penalty = (expert_q ** 2).mean()
            loss = loss + self.gradient_penalty_weight * grad_penalty

        # ---- reward_reg: separate MSE on recovered reward ----
        svo_reg_loss = torch.tensor(0.0, device=self.device)

        if self.use_svo and self.svo_mode in ('reward_reg', 'reward_reg_reweight'):
            # Recovered reward: r̂(s,a) = Q(s,a) − γ(1−d)V(s')
            r_recovered = expert_q - gamma_v_next

            # Target: λ · R_SVO
            svo_target = svo_shift  # already = λ * r_svo

            svo_reg_loss = ((r_recovered - svo_target) ** 2).mean()
            loss = loss + svo_reg_loss

        # ---- Logging ----
        abs_q_mean       = expert_q.abs().mean().item()
        abs_gamma_v_mean = gamma_v_next.abs().mean().item()

        info = {
            'loss':              loss.item(),
            'expert_q_mean':     expert_q.mean().item(),
            'expert_next_v_mean': expert_next_v.mean().item(),
            'grad_penalty':      grad_penalty.item(),
            'abs_expert_q_mean': abs_q_mean,
            'abs_gamma_v_mean':  abs_gamma_v_mean,
        }

        if self.use_svo and r_svo_raw is not None:
            abs_svo_shift_mean = svo_shift.abs().mean().item() if torch.is_tensor(svo_shift) else 0.0
            shift_to_q_ratio   = abs_svo_shift_mean / (abs_q_mean + 1e-8)

            info['svo_raw_mean']         = r_svo_raw.mean().item()
            info['svo_raw_std']          = r_svo_raw.std().item()
            info['svo_shift_mean']       = svo_shift.mean().item() if torch.is_tensor(svo_shift) else 0.0
            info['svo_shift_std']        = svo_shift.std().item()  if torch.is_tensor(svo_shift) else 0.0
            info['svo_shift_min']        = svo_shift.min().item()  if torch.is_tensor(svo_shift) else 0.0
            info['svo_shift_max']        = svo_shift.max().item()  if torch.is_tensor(svo_shift) else 0.0
            info['abs_svo_shift_mean']   = abs_svo_shift_mean
            info['svo_shift_to_q_ratio'] = shift_to_q_ratio

        if self.use_svo and self.svo_mode in ('reward_reg', 'reward_reg_reweight'):
            r_recovered_detached = (expert_q - gamma_v_next).detach()
            info['svo_reg_loss']          = svo_reg_loss.item()
            info['recovered_reward_mean'] = r_recovered_detached.mean().item()
            info['recovered_reward_std']  = r_recovered_detached.std().item()

        if self.use_svo and self.svo_mode in ('reweight', 'reward_reg_reweight'):
            r_svo_batch = self._compute_svo_reward(
                expert_r_selfs, expert_r_globals).to(self.device)
            info['svo_raw_mean'] = r_svo_batch.mean().item()
            info['svo_raw_std']  = r_svo_batch.std().item()

        if len(learner_states) > 0:
            info['learner_q_mean']     = learner_q.mean().item()
            info['learner_next_v_mean'] = learner_next_v.mean().item()

        return loss, info

    # ==================================================================
    # Update step
    # ==================================================================

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Perform one gradient update."""

        expert_batch_size  = int(batch_size * self.replay_ratio)
        learner_batch_size = batch_size - expert_batch_size

        # --- Sample expert data ---
        if self.use_svo and self.svo_mode in ('reweight', 'reward_reg_reweight') and self._expert_svo_weights is not None:
            expert_batch = self.expert_buffer.sample_weighted(
                expert_batch_size, self._expert_svo_weights)
        else:
            expert_batch = self.expert_buffer.sample(expert_batch_size)

        (expert_states, expert_actions, _, expert_next_states,
         expert_dones, expert_r_selfs, expert_r_globals) = expert_batch

        expert_states      = expert_states.to(self.device)
        expert_actions     = expert_actions.to(self.device)
        expert_next_states = expert_next_states.to(self.device)
        expert_dones       = expert_dones.to(self.device)
        expert_r_selfs     = expert_r_selfs.to(self.device)
        expert_r_globals   = expert_r_globals.to(self.device)

        # --- Sample learner data ---
        if len(self.learner_buffer) >= learner_batch_size:
            learner_batch = self.learner_buffer.sample(learner_batch_size)
            (learner_states, learner_actions, _, learner_next_states,
             learner_dones, _, _) = learner_batch

            learner_states      = learner_states.to(self.device)
            learner_actions     = learner_actions.to(self.device)
            learner_next_states = learner_next_states.to(self.device)
            learner_dones       = learner_dones.to(self.device)
        else:
            learner_states      = torch.empty(0, self.state_dim).to(self.device)
            learner_actions     = torch.empty(0, dtype=torch.long).to(self.device)
            learner_next_states = torch.empty(0, self.state_dim).to(self.device)
            learner_dones       = torch.empty(0).to(self.device)

        # --- Compute loss ---
        loss, info = self.compute_iq_loss(
            expert_states, expert_actions, expert_next_states, expert_dones,
            expert_r_selfs, expert_r_globals,
            learner_states, learner_actions, learner_next_states, learner_dones,
        )

        # --- Optimise ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self._soft_update_target()

        # --- Logging ---
        self.losses.append(info['loss'])
        self.expert_q_values.append(info['expert_q_mean'])
        self.abs_q_values.append(info['abs_expert_q_mean'])
        if 'learner_q_mean' in info:
            self.learner_q_values.append(info['learner_q_mean'])
        if self.use_svo and 'svo_shift_to_q_ratio' in info:
            self.svo_ratios.append(info['svo_shift_to_q_ratio'])
            self.abs_svo_shifts.append(info['abs_svo_shift_mean'])

        return info

    # ==================================================================
    # Utilities (unchanged from original)
    # ==================================================================

    def _soft_update_target(self):
        for tp, p in zip(self.q_target.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        state_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state_t).argmax(dim=1).item()

    def collect_learner_rollout(self, num_steps: int = 1000, epsilon: float = 0.1):
        state, _ = self.env.reset()
        for _ in range(num_steps):
            action = self.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            r_self   = info.get('rewards/component_self', 0.0)
            r_global = info.get('rewards/component_global', 0.0)
            self.learner_buffer.add(state, action, reward, next_state, done,
                                    r_self, r_global)
            state = next_state
            if done:
                state, _ = self.env.reset()

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        episode_rewards, episode_lengths = [], []
        collision_count = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward, ep_len, done = 0, 0, False
            while not done:
                action = self.select_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_len += 1
                state = next_state
                if terminated and hasattr(self.env.unwrapped, 'vehicle'):
                    if self.env.unwrapped.vehicle.crashed:
                        collision_count += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
        return {
            'mean_reward':   np.mean(episode_rewards),
            'std_reward':    np.std(episode_rewards),
            'mean_length':   np.mean(episode_lengths),
            'collision_rate': collision_count / num_episodes,
        }

    def save(self, path: str):
        torch.save({
            'q_network':  self.q_network.state_dict(),
            'q_target':   self.q_target.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'losses':     self.losses,
            'expert_q_values':  self.expert_q_values,
            'learner_q_values': self.learner_q_values,
            # SVO config
            'use_svo':       self.use_svo,
            'svo_mode':      self.svo_mode,
            'svo_alpha':     self.svo_alpha,
            'svo_lambda':    self.svo_lambda,
            'normalize_svo': self.normalize_svo,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt['q_network'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.losses           = ckpt.get('losses', [])
        self.expert_q_values  = ckpt.get('expert_q_values', [])
        self.learner_q_values = ckpt.get('learner_q_values', [])
        print(f"Loaded model from {path}")