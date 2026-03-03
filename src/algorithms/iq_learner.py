"""
IQ-Learn implementation based on "IQ-Learn: Inverse soft-Q Learning for Imitation" (Garg et al., 2021)

Supports optional SVO regularization:
    When use_svo=True, the Bellman target for expert data is shifted by
    λ · R_SVO(s, a | α_target), biasing the learned reward toward the
    desired social preference.
    When use_svo=False (default), behaviour is identical to standard IQ-Learn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import gymnasium as gym


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
            5-element: (s, a, r, s', done)                          — minimal
            6-element: (s, a, r, s', done, crashed)                 — legacy
            8-element: (s, a, r, s', done, crashed, r_self, r_global) — current

        The buffer always stores 7 fields: (s, a, r, s', done, r_self, r_global).
        'crashed' is dropped since IQ-Learn does not use it.
        """
        for transition in trajectory:
            n = len(transition)
            if n >= 8:
                # Current format: (s, a, r, s', done, crashed, r_self, r_global)
                # Skip [5] (crashed), use [6] and [7]
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                    float(transition[6]), float(transition[7]),
                )
            elif n == 6:
                # Legacy format: (s, a, r, s', done, crashed)
                # No SVO components available
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                )
            else:
                # Minimal format: (s, a, r, s', done)
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                )

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor(np.array([x[0].flatten() for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3].flatten() for x in batch]))
        dones = torch.FloatTensor(np.array([x[4] for x in batch]))
        r_selfs = torch.FloatTensor(np.array([x[5] for x in batch]))
        r_globals = torch.FloatTensor(np.array([x[6] for x in batch]))

        return states, actions, rewards, next_states, dones, r_selfs, r_globals

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Q-Network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
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


class IQLearnTrainer:
    """
    IQ-Learn Trainer for learning from expert demonstrations.

    When ``use_svo=True``, the expert Bellman target is shifted by
    λ · (cos(α) · r_self + sin(α) · r_global) so the Q-function
    preferentially fits transitions aligned with α_target.
    """

    def __init__(self,
                 env: gym.Env,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # IQ-Learn parameters
                 method: str = 'value',
                 loss_type: str = 'v0',
                 regularize_weight: float = 1.0,
                 temperature: float = 1.0,
                 # Stabilization
                 gradient_penalty_weight: float = 0.1,
                 replay_ratio: float = 0.5,
                 # SVO regularization
                 use_svo: bool = False,
                 svo_alpha: float = 0.0,
                 svo_lambda: float = 1.0,
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

        # SVO regularization
        self.use_svo = use_svo
        self.svo_alpha = svo_alpha          # target angle in radians
        self.svo_lambda = svo_lambda        # regularization strength
        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)

        if self.use_svo:
            print(f"[SVO-IQ] SVO regularization ENABLED")
            print(f"[SVO-IQ]   α_target = {np.degrees(svo_alpha):.1f}°  ({svo_alpha:.4f} rad)")
            print(f"[SVO-IQ]   λ        = {svo_lambda}")
        else:
            print(f"[IQ-Learn] Standard IQ-Learn (no SVO regularization)")

        # Q-networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q_target = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Buffers
        self.expert_buffer = ReplayBuffer()
        self.learner_buffer = ReplayBuffer()

        # Logging
        self.losses = []
        self.expert_q_values = []
        self.learner_q_values = []

    def load_expert_demonstrations(self, demonstrations: List[List[Tuple]]):
        """
        Load expert demonstrations into expert buffer.

        Rewards are stored but IGNORED in reward-free IQ-Learn.
        r_self and r_global are used only when use_svo=True.
        """
        print(f"Loading {len(demonstrations)} expert demonstrations")

        has_svo_components = False
        if demonstrations and demonstrations[0]:
            has_svo_components = len(demonstrations[0][0]) >= 8

        if self.use_svo and not has_svo_components:
            print(
                "[SVO-IQ] WARNING: SVO regularization is enabled but "
                "demonstrations do not contain r_self / r_global components. "
                "SVO reward shaping will use zeros (no effect)."
            )

        for traj in demonstrations:
            self.expert_buffer.add_trajectory(traj)
        print(f"Expert buffer size: {len(self.expert_buffer)}")

    def _compute_svo_reward(self, r_selfs: torch.Tensor, r_globals: torch.Tensor) -> torch.Tensor:
        """
        Compute R_SVO = cos(α_target) · r_self + sin(α_target) · r_global

        Returns a [batch, 1] tensor.
        """
        r_svo = self.cos_alpha * r_selfs + self.sin_alpha * r_globals
        return r_svo.unsqueeze(1)

    def soft_value_function(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute soft value function: V(s) = temperature * log(sum_a exp(Q(s,a) / temperature))
        """
        q_values = self.q_network(states)
        v = self.temperature * torch.logsumexp(q_values / self.temperature, dim=1, keepdim=True)
        return v

    def soft_value_function_target(self, states: torch.Tensor) -> torch.Tensor:
        """Compute soft value using target network"""
        with torch.no_grad():
            q_values = self.q_target(states)
            v = self.temperature * torch.logsumexp(q_values / self.temperature, dim=1, keepdim=True)
        return v

    def compute_iq_loss(self,
                        expert_states: torch.Tensor,
                        expert_actions: torch.Tensor,
                        expert_next_states: torch.Tensor,
                        expert_dones: torch.Tensor,
                        expert_r_selfs: torch.Tensor,
                        expert_r_globals: torch.Tensor,
                        learner_states: torch.Tensor,
                        learner_actions: torch.Tensor,
                        learner_next_states: torch.Tensor,
                        learner_dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute IQ-Learn loss, optionally with SVO-shifted Bellman target.

        When use_svo=True:
            expert_target = λ·R_SVO(s,a) + γ(1-done)V(s')
        When use_svo=False:
            expert_target = γ(1-done)V(s')       (standard IQ-Learn)
        """

        # Expert Q-values
        expert_q = self.q_network(expert_states).gather(1, expert_actions.unsqueeze(1))
        expert_next_v = self.soft_value_function_target(expert_next_states)

        # Learner Q-values
        if len(learner_states) > 0:
            learner_q = self.q_network(learner_states).gather(1, learner_actions.unsqueeze(1))
            learner_next_v = self.soft_value_function_target(learner_next_states)
        else:
            learner_q = torch.zeros_like(expert_q)
            learner_next_v = torch.zeros_like(expert_next_v)

        # ---- SVO reward shaping (only for expert data) ----
        if self.use_svo:
            r_svo = self._compute_svo_reward(expert_r_selfs, expert_r_globals)
            r_svo = r_svo.to(self.device)
            svo_shift = self.svo_lambda * r_svo
        else:
            svo_shift = 0.0  # scalar zero, broadcasts cleanly

        # ---- Loss computation ----
        if self.loss_type == 'v0':
            # Expert Bellman residual (with optional SVO shift)
            expert_target = svo_shift + self.gamma * (1 - expert_dones.unsqueeze(1)) * expert_next_v
            expert_bellman_residual = expert_q - expert_target

            if len(learner_states) > 0:
                # Learner target is NOT shifted by SVO — learner data comes
                # from the current policy, not the demonstration dataset.
                learner_target = self.gamma * (1 - learner_dones.unsqueeze(1)) * learner_next_v
                learner_bellman_residual = learner_q - learner_target

                expert_loss = (expert_bellman_residual ** 2).mean()
                learner_loss = 0.5 * F.relu(learner_bellman_residual).mean()

                loss = expert_loss + self.regularize_weight * learner_loss
            else:
                loss = (expert_bellman_residual ** 2).mean()

            grad_penalty = (expert_q ** 2).mean()
            loss = loss + self.gradient_penalty_weight * grad_penalty

        elif self.loss_type == 'v1':
            # Direct Q-value matching with SVO shift
            # Maximize (Q - λ·R_SVO) for expert -> encourages Q to be high
            # specifically when R_SVO is high.
            expert_loss = -(expert_q - svo_shift).mean()

            if len(learner_states) > 0:
                learner_loss = learner_q.mean()
                loss = expert_loss + self.regularize_weight * learner_loss
            else:
                loss = expert_loss

            grad_penalty = (expert_q ** 2).mean()
            loss = loss + self.gradient_penalty_weight * grad_penalty

        # ---- Logging info ----
        info = {
            'loss': loss.item(),
            'expert_q_mean': expert_q.mean().item(),
            'expert_next_v_mean': expert_next_v.mean().item(),
            'grad_penalty': grad_penalty.item(),
        }

        if self.use_svo:
            info['svo_reward_mean'] = r_svo.mean().item()

        if len(learner_states) > 0:
            info['learner_q_mean'] = learner_q.mean().item()
            info['learner_next_v_mean'] = learner_next_v.mean().item()

        return loss, info

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Perform one update step"""

        expert_batch_size = int(batch_size * self.replay_ratio)
        learner_batch_size = batch_size - expert_batch_size

        # Sample from expert buffer (now returns 7 elements)
        expert_batch = self.expert_buffer.sample(expert_batch_size)
        (expert_states, expert_actions, _, expert_next_states,
         expert_dones, expert_r_selfs, expert_r_globals) = expert_batch

        expert_states = expert_states.to(self.device)
        expert_actions = expert_actions.to(self.device)
        expert_next_states = expert_next_states.to(self.device)
        expert_dones = expert_dones.to(self.device)
        expert_r_selfs = expert_r_selfs.to(self.device)
        expert_r_globals = expert_r_globals.to(self.device)

        # Sample from learner buffer (if available)
        if len(self.learner_buffer) >= learner_batch_size:
            learner_batch = self.learner_buffer.sample(learner_batch_size)
            (learner_states, learner_actions, _, learner_next_states,
             learner_dones, _, _) = learner_batch

            learner_states = learner_states.to(self.device)
            learner_actions = learner_actions.to(self.device)
            learner_next_states = learner_next_states.to(self.device)
            learner_dones = learner_dones.to(self.device)
        else:
            learner_states = torch.empty(0, self.state_dim).to(self.device)
            learner_actions = torch.empty(0, dtype=torch.long).to(self.device)
            learner_next_states = torch.empty(0, self.state_dim).to(self.device)
            learner_dones = torch.empty(0).to(self.device)

        # Compute loss
        loss, info = self.compute_iq_loss(
            expert_states, expert_actions, expert_next_states, expert_dones,
            expert_r_selfs, expert_r_globals,
            learner_states, learner_actions, learner_next_states, learner_dones,
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target()

        # Logging
        self.losses.append(info['loss'])
        self.expert_q_values.append(info['expert_q_mean'])
        if 'learner_q_mean' in info:
            self.learner_q_values.append(info['learner_q_mean'])

        return info

    def _soft_update_target(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def collect_learner_rollout(self, num_steps: int = 1000, epsilon: float = 0.1):
        """Collect rollout using current learned policy"""
        state, _ = self.env.reset()

        for _ in range(num_steps):
            action = self.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store SVO components from the wrapper if available
            r_self = info.get('rewards/component_self', 0.0)
            r_global = info.get('rewards/component_global', 0.0)

            self.learner_buffer.add(state, action, reward, next_state, done,
                                    r_self, r_global)

            state = next_state
            if done:
                state, _ = self.env.reset()

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy"""
        episode_rewards = []
        episode_lengths = []
        collision_count = 0

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1
                state = next_state

                if terminated and hasattr(self.env.unwrapped, 'vehicle'):
                    if self.env.unwrapped.vehicle.crashed:
                        collision_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'collision_rate': collision_count / num_episodes,
        }

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'expert_q_values': self.expert_q_values,
            'learner_q_values': self.learner_q_values,
            # SVO config (so we know what was used when loading)
            'use_svo': self.use_svo,
            'svo_alpha': self.svo_alpha,
            'svo_lambda': self.svo_lambda,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.losses = checkpoint.get('losses', [])
        self.expert_q_values = checkpoint.get('expert_q_values', [])
        self.learner_q_values = checkpoint.get('learner_q_values', [])
        print(f"Loaded model from {path}")