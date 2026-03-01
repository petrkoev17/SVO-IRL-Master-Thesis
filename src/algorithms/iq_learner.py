"""
IQ-Learn implementation based on "IQ-Learn: Inverse soft-Q Learning for Imitation" (Garg et al., 2021)
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
    Replay Buffer for expert demonstrations and learner experience
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def add_trajectory(self, trajectory: List[Tuple]):
        """
        Add entire trajectory to the buffer
        """
        for transition in trajectory:
            self.add(*transition)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor(np.array([x[0].flatten() for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3].flatten() for x in batch]))
        dones = torch.FloatTensor(np.array([x[4] for x in batch]))

        return states, actions, rewards, next_states, dones

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
    IQ-Learn Trainer for learning from expert demonstrations
    """

    def __init__(self,
                 env: gym.Env,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005, # soft update parameter
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # IQ-Learn parameters
                 method: str = 'value', # 'value' or 'q'
                 loss_type: str = 'v0', # 'v0' or 'v1'
                 regularize_weight: float = 1.0,
                 temperature: float = 1.0,
                 # Stabilization
                 gradient_penalty_weight: float = 0.1,
                 replay_ratio: float = 0.5, # Expert/Learner ratio
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
        Load expert demonstrations into expert buffer
        Args:
            demonstrations: List of trajectories, where each trajectory is a list of
                          (state, action, reward, next_state, done) tuples

        Rewards are stored but IGNORED in reward-free IQ-Learn
        """

        print(f"Loading {len(demonstrations)} expert demonstrations")
        for traj in demonstrations:
            self.expert_buffer.add_trajectory(traj)
        print(f"Expert buffer size: {len(self.expert_buffer)}")

    def soft_value_function(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute soft value function: V(s) = temperature * log(sum_a exp(Q(s,a) / temperature))
        Soft maximum over actions
        """

        q_values = self.q_network(states)
        v = self.temperature * torch.logsumexp(q_values / self.temperature, dim=1, keepdim=True)

        return v

    def soft_value_function_target(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute soft value using target network
        """
        with torch.no_grad():
            q_values = self.q_target(states)
            v = self.temperature * torch.logsumexp(q_values / self.temperature, dim=1, keepdim=True)

        return v

    def compute_iq_loss(self,
                        expert_states: torch.Tensor,
                        expert_actions: torch.Tensor,
                        expert_next_states: torch.Tensor,
                        expert_dones: torch.Tensor,
                        learner_states: torch.Tensor,
                        learner_actions: torch.Tensor,
                        learner_next_states: torch.Tensor,
                        learner_dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute IQ-Learn loss

        The core idea: Expert Q-values should be higher than learner Q-values,
        and the Bellman equation should hold for expert data
        """

        # Get Q-Values for expert demonstrations
        expert_q = self.q_network(expert_states).gather(1, expert_actions.unsqueeze(1))
        expert_next_v = self.soft_value_function_target(expert_next_states)

        # Get Q-Values for learner samples
        if len(learner_states) > 0:
            learner_q = self.q_network(learner_states).gather(1, learner_actions.unsqueeze(1))
            learner_next_v = self.soft_value_function_target(learner_next_states)
        else:
            learner_q = torch.zeros_like(expert_q)
            learner_next_v = torch.zeros_like(expert_next_v)

        # Compute IQ loss based on variant
        if self.loss_type == 'v0':
            # Original IQ-Learn loss
            # Expert should satisfy: Q(s,a) ≈ γV(s')
            # Learner should have: Q(s,a) ≤ γV(s')

            expert_target = self.gamma * (1 - expert_dones.unsqueeze(1)) * expert_next_v
            expert_bellman_residual = expert_q - expert_target

            if len(learner_states) > 0:
                learner_target = self.gamma * (1 - learner_dones.unsqueeze(1)) * learner_next_v
                learner_bellman_residual = learner_q - learner_target

                # Chi-squared divergence term for expert
                expert_loss = (expert_bellman_residual ** 2).mean()

                # Regularization: push Q-values down
                learner_loss = 0.5 * F.relu(learner_bellman_residual).mean()

                loss = expert_loss + self.regularize_weight * learner_loss

            else:
                loss = (expert_bellman_residual ** 2).mean()

            grad_penalty = (expert_q ** 2).mean()
            loss = loss + self.gradient_penalty_weight * grad_penalty

        elif self.loss_type == 'v1':
            # Alternative: Direct Q-value matching with regularization
            # Maximize expert Q-values, minimize learner Q-values

            expert_loss = -expert_q.mean()

            if len(learner_states) > 0:
                learner_loss = learner_q.mean()
                loss = expert_loss + self.regularize_weight * learner_loss
            else:
                loss = expert_loss

            grad_penalty = (expert_q ** 2).mean()
            loss = loss + self.gradient_penalty_weight * grad_penalty

        # Logging info
        info = {
            'loss': loss.item(),
            'expert_q_mean': expert_q.mean().item(),
            'expert_next_v_mean': expert_next_v.mean().item(),
            'grad_penalty': grad_penalty.item(),
        }

        if len(learner_states) > 0:
            info['learner_q_mean'] = learner_q.mean().item()
            info['learner_next_v_mean'] = learner_next_v.mean().item()

        return loss, info

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Perform one update step
        """

        # Split batch between expert and learner
        expert_batch_size = int(batch_size * self.replay_ratio)
        learner_batch_size = batch_size - expert_batch_size

        # Sample from expert buffer
        expert_batch = self.expert_buffer.sample(expert_batch_size)
        expert_states, expert_actions, _, expert_next_states, expert_dones = expert_batch

        expert_states = expert_states.to(self.device)
        expert_actions = expert_actions.to(self.device)
        expert_next_states = expert_next_states.to(self.device)
        expert_dones = expert_dones.to(self.device)

        # Sample from learner buffer (if available)
        if len(self.learner_buffer) >= learner_batch_size:
            learner_batch = self.learner_buffer.sample(learner_batch_size)
            learner_states, learner_actions, _, learner_next_states, learner_dones = learner_batch

            learner_states = learner_states.to(self.device)
            learner_actions = learner_actions.to(self.device)
            learner_next_states = learner_next_states.to(self.device)
            learner_dones = learner_dones.to(self.device)
        else:
            # Empty tensors
            learner_states = torch.empty(0, self.state_dim).to(self.device)
            learner_actions = torch.empty(0, dtype=torch.long).to(self.device)
            learner_next_states = torch.empty(0, self.state_dim).to(self.device)
            learner_dones = torch.empty(0).to(self.device)


        # Compute loss
        loss, info = self.compute_iq_loss(
            expert_states, expert_actions, expert_next_states, expert_dones,
            learner_states, learner_actions, learner_next_states, learner_dones
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
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
        """
        Soft update target network parameters
        """

        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy
        """
        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def collect_learner_rollout(self, num_steps: int = 1000, epsilon: float = 0.1):
        """
        Collect rollout using current learned policy
        """

        state, _ = self.env.reset()

        for _ in range(num_steps):
            action = self.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.learner_buffer.add(state, action, reward, next_state, done)

            state = next_state
            if done:
                state, _ = self.env.reset()

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy
        """

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
        """
        Save model checkpoint
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'expert_q_values': self.expert_q_values,
            'learner_q_values': self.learner_q_values,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.losses = checkpoint.get('losses', [])
        self.expert_q_values = checkpoint.get('expert_q_values', [])
        self.learner_q_values = checkpoint.get('learner_q_values', [])
        print(f"Loaded model from {path}")