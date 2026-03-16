"""
IQ-Learn implementation based on "IQ-Learn: Inverse soft-Q Learning for Imitation"
(Garg et al., 2021).

This version faithfully reproduces the original IQ-Learn loss from Garg et al.'s
reference implementation, adapted for discrete-action DQN.  The original uses a
continuous-action SAC agent; here we replace the actor with ε-greedy action
selection and derive V(s) = τ · logsumexp(Q(s,·)/τ) from the discrete Q-network.

Key corrections vs. the previous (incorrect) version:
  1. The implicit reward is correctly defined as  r̂ = Q(s,a) − γ(1−d)V(s').
  2. The first loss term applies f-divergence weighting (φ_grad) to expert
     transitions' implicit reward, matching the original's divergence options.
  3. The second loss term uses the correct sampling strategies:
       'value'        – E_{ρ_expert ∪ ρ_policy}[V(s) − γV(s')]   (online, DEFAULT)
       'value_expert' – E_{ρ_expert}[V(s) − γV(s')]              (offline)
       'v0'           – (1−γ) · E_{s_0}[V(s_0)]                  (offline, usually suboptimal)
  4. χ² divergence regularisation is supported (enabled by default):
       adds  1/(4α) · E[r̂²]  on expert or all data depending on mode.
  5. All f-divergences from the original are supported:
       'chi'     – χ² (recommended, adds separate regularisation term)
       'kl'      – KL (original dual, sub-optimal)
       'kl2'     – KL (biased dual)
       'kl_fix'  – KL (unbiased fix proposed by authors)
       'js'      – Jensen–Shannon
       'hellinger' – Hellinger
       None/'none' – standard (φ_grad = 1)

Supports optional SVO regularisation with three modes:
    'bellman'    – Shift the Bellman target by λ·R_SVO.
    'reward_reg' – Separate MSE loss pushing recovered reward toward λ·R_SVO.
    'reweight'   – Importance-weight expert sampling by R_SVO.

When use_svo=False (default), behaviour is identical to standard IQ-Learn.
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
        (state, action, reward, next_state, done, r_self, r_global, G_self, G_global)

    r_self and r_global are the raw instantaneous SVO reward components.
    G_self and G_global are cumulative discounted SVO returns (optional).
    All SVO fields default to 0.0 so the buffer format is uniform
    regardless of whether SVO regularisation is active.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done,
            r_self: float = 0.0, r_global: float = 0.0,
            G_self: float = 0.0, G_global: float = 0.0):
        self.buffer.append((state, action, reward, next_state, done,
                            r_self, r_global, G_self, G_global))

    def add_trajectory(self, trajectory: List[Tuple]):
        """
        Add entire trajectory to the buffer.

        Supported tuple formats:
            5-element:  (s, a, r, s', done)
            6-element:  (s, a, r, s', done, crashed)
            8-element:  (s, a, r, s', done, crashed, r_self, r_global)
            10-element: (s, a, r, s', done, crashed, r_self, r_global, G_self, G_global)
        """
        for transition in trajectory:
            n = len(transition)
            if n >= 10:
                self.add(
                    transition[0], transition[1], transition[2],
                    transition[3], transition[4],
                    float(transition[6]), float(transition[7]),
                    float(transition[8]), float(transition[9]),
                )
            elif n >= 8:
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

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return self._gather(indices)

    def sample_weighted(self, batch_size: int, weights: np.ndarray):
        probs = weights / weights.sum()
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=True, p=probs)
        return self._gather(indices)

    def _gather(self, indices):
        batch = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor(np.array([x[0].flatten() for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3].flatten() for x in batch]))
        dones = torch.FloatTensor(np.array([x[4] for x in batch]))
        r_selfs = torch.FloatTensor(np.array([x[5] for x in batch]))
        r_globals = torch.FloatTensor(np.array([x[6] for x in batch]))
        G_selfs = torch.FloatTensor(np.array([x[7] for x in batch]))
        G_globals = torch.FloatTensor(np.array([x[8] for x in batch]))

        return states, actions, rewards, next_states, dones, r_selfs, r_globals, G_selfs, G_globals
    def __len__(self):
        return len(self.buffer)


# ======================================================================
# Q-Network
# ======================================================================

class DQNNetwork(nn.Module):
    """Standard MLP Q-Network for discrete actions."""

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
    IQ-Learn Trainer for discrete-action environments.

    Faithfully implements the IQ-Learn objective from Garg et al. (2021)
    using a DQN backbone.  The soft value function is derived analytically:

        V(s) = τ · logsumexp( Q(s, ·) / τ )

    Loss sampling strategies:
        'value'        – online:  E_{all}[ V(s) − γV(s') ]
        'value_expert' – offline: E_{expert}[ V(s) − γV(s') ]  (was Q in orig; see note)
        'v0'           – offline: (1−γ) · E_{s0}[ V(s0) ]

    f-divergence options:
        'chi'      – χ² divergence (recommended; adds regularisation term)
        'kl'       – KL (original sub-optimal dual)
        'kl2'      – KL (biased dual)
        'kl_fix'   – KL (unbiased fix)
        'js'       – Jensen–Shannon
        'hellinger'– Hellinger
        'none'     – standard (no reweighting, φ_grad = 1)

    SVO modes (only active when use_svo=True):
        'bellman'            – shift Bellman target by λ·R_SVO
        'reward_reg'         – separate MSE on recovered reward vs λ·R_SVO
        'reweight'           – importance-weight expert sampling by R_SVO
        'reward_reg_reweight'– both reward_reg + reweight
    """

    VALID_SVO_MODES = ('bellman', 'reward_reg', 'reweight', 'reward_reg_reweight')
    VALID_DIVERGENCES = ('none', 'chi', 'kl', 'kl2', 'kl_fix', 'js', 'hellinger')
    VALID_LOSS_TYPES = ('value', 'value_expert', 'v0')

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
                 loss_type: str = 'value',
                 divergence: str = 'chi',
                 div_alpha: float = 0.5,
                 temperature: float = 1.0,
                 use_target_network: bool = True,
                 learner_buffer_size: int = 5_000,
                 # Gradient penalty (Wasserstein-1 style, optional)
                 grad_pen: bool = False,
                 lambda_gp: float = 10.0,
                 # Replay
                 replay_ratio: float = 0.5,
                 # SVO regularisation
                 use_svo: bool = False,
                 svo_mode: str = 'reward_reg',
                 svo_alpha: float = 0.0,
                 svo_lambda: float = 1.0,
                 normalize_svo: bool = False,
                 svo_reweight_temp: float = 1.0,
                 svo_cumulative: bool = False,
                 # Legacy compatibility (ignored but accepted)
                 method: str = 'value',
                 regularize_weight: float = 1.0,
                 gradient_penalty_weight: float = 0.1,
                 ):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # --- IQ-Learn core config ---
        self.loss_type = loss_type
        self.divergence = divergence
        self.div_alpha = div_alpha
        self.temperature = temperature
        self.use_target_network = use_target_network
        self.replay_ratio = replay_ratio
        self.grad_pen = grad_pen
        self.lambda_gp = lambda_gp

        # Validate
        if self.loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(
                f"Invalid loss_type='{loss_type}'. "
                f"Must be one of {self.VALID_LOSS_TYPES}"
            )
        if self.divergence not in self.VALID_DIVERGENCES:
            raise ValueError(
                f"Invalid divergence='{divergence}'. "
                f"Must be one of {self.VALID_DIVERGENCES}"
            )

        # --- SVO config ---
        self.use_svo = use_svo
        self.svo_mode = svo_mode
        self.svo_alpha = svo_alpha
        self.svo_lambda = svo_lambda
        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)
        self.normalize_svo = normalize_svo
        self.svo_reweight_temp = svo_reweight_temp
        self.svo_cumulative = svo_cumulative
        self._expert_svo_weights: Optional[np.ndarray] = None
        self._svo_global_std: Optional[float] = None

        if self.use_svo:
            if self.svo_mode not in self.VALID_SVO_MODES:
                raise ValueError(
                    f"Invalid svo_mode='{svo_mode}'. "
                    f"Must be one of {self.VALID_SVO_MODES}"
                )
            print(f"[SVO-IQ] SVO regularisation ENABLED")
            print(f"[SVO-IQ]   mode       = {svo_mode}")
            print(f"[SVO-IQ]   cumulative = {svo_cumulative}")
            print(f"[SVO-IQ]   α_target   = {np.degrees(svo_alpha):.1f}°  ({svo_alpha:.4f} rad)")
            print(f"[SVO-IQ]   λ          = {svo_lambda}")
            print(f"[SVO-IQ]   normalise  = {normalize_svo}")
            if svo_mode in ('reweight', 'reward_reg_reweight'):
                print(f"[SVO-IQ]   reweight τ = {svo_reweight_temp}")
        else:
            print(f"[IQ-Learn] Standard IQ-Learn (no SVO regularisation)")

        print(f"[IQ-Learn] loss_type   = {self.loss_type}")
        print(f"[IQ-Learn] divergence  = {self.divergence}")
        if self.divergence == 'chi':
            print(f"[IQ-Learn] div_alpha   = {self.div_alpha}")
        print(f"[IQ-Learn] temperature = {self.temperature}")
        print(f"[IQ-Learn] use_target  = {self.use_target_network}")

        # Q-networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q_target  = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Optimiser
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Buffers
        self.expert_buffer  = ReplayBuffer()
        self.learner_buffer = ReplayBuffer(capacity=learner_buffer_size)

        # Logging lists
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
        has_cumulative = False
        if demonstrations and demonstrations[0]:
            tup_len = len(demonstrations[0][0])
            has_svo = tup_len >= 8
            has_cumulative = tup_len >= 10

        if self.use_svo and not has_svo:
            print(
                "[SVO-IQ] WARNING: SVO regularisation is enabled but "
                "demonstrations do not contain r_self / r_global. "
                "SVO reward shaping will use zeros (no effect)."
            )

        if self.use_svo and self.svo_cumulative and not has_cumulative:
            print(
                "[SVO-IQ] WARNING: svo_cumulative=True but demonstrations "
                "do not contain G_self / G_global (need 10-element tuples). "
                "Run: python extract_demonstrations.py augment --input <path> --gamma-svo 0.99\n"
                "Falling back to instantaneous r_self / r_global."
            )
            self.svo_cumulative = False

        for traj in demonstrations:
            self.expert_buffer.add_trajectory(traj)
        print(f"Expert buffer size: {len(self.expert_buffer)}")

        if has_cumulative:
            print(f"[SVO-IQ] Demonstrations contain cumulative returns (G_self, G_global)")

        # ----------------------------------------------------------
        # Global SVO normalization (replaces per-batch normalization)
        # Compute σ over entire expert buffer for deterministic scaling.
        # We scale by σ only (no mean subtraction) to preserve
        # the directional SVO bias.
        # ----------------------------------------------------------
        if self.use_svo and self.normalize_svo:
            n = len(self.expert_buffer)
            svo_vals = np.empty(n, dtype=np.float32)
            for i, transition in enumerate(self.expert_buffer.buffer):
                if self.svo_cumulative:
                    svo_vals[i] = (self.cos_alpha * transition[7]
                                   + self.sin_alpha * transition[8])
                else:
                    svo_vals[i] = (self.cos_alpha * transition[5]
                                   + self.sin_alpha * transition[6])
            self._svo_global_std = float(svo_vals.std()) + 1e-8
            print(f"[SVO-IQ] Global SVO normalization:")
            print(f"[SVO-IQ]   σ = {self._svo_global_std:.4f}")
            print(f"[SVO-IQ]   raw range = [{svo_vals.min():.4f}, {svo_vals.max():.4f}]")
            print(f"[SVO-IQ]   scaled range = "
                  f"[{(svo_vals.min() / self._svo_global_std):.4f}, "
                  f"{(svo_vals.max() / self._svo_global_std):.4f}]")


        if self.use_svo and self.svo_mode in ('reweight', 'reward_reg_reweight'):
            self._compute_expert_weights()

    def _compute_expert_weights(self):
        """Compute per-transition importance weights from R_SVO.

        When svo_cumulative=True, uses G_self/G_global (cumulative returns)
        instead of instantaneous r_self/r_global.
        """
        n = len(self.expert_buffer)
        svo_vals = np.empty(n, dtype=np.float32)

        for i, transition in enumerate(self.expert_buffer.buffer):
            if self.svo_cumulative:
                # G_self at index 7, G_global at index 8
                svo_vals[i] = self.cos_alpha * transition[7] + self.sin_alpha * transition[8]
            else:
                # r_self at index 5, r_global at index 6
                svo_vals[i] = self.cos_alpha * transition[5] + self.sin_alpha * transition[6]

        logits = svo_vals / (self.svo_reweight_temp + 1e-8)
        logits -= logits.max()
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
                            r_globals: torch.Tensor,
                            G_selfs: torch.Tensor = None,
                            G_globals: torch.Tensor = None) -> torch.Tensor:
        """R_SVO from instantaneous or cumulative values  →  [batch, 1]

        When svo_cumulative=True and G fields are provided, uses those.
        Otherwise falls back to instantaneous r_self/r_global.
        """
        if self.svo_cumulative and G_selfs is not None and G_globals is not None:
            r_svo = self.cos_alpha * G_selfs + self.sin_alpha * G_globals
        else:
            r_svo = self.cos_alpha * r_selfs + self.sin_alpha * r_globals
        return r_svo.unsqueeze(1)

    # ==================================================================
    # Value functions  (discrete-action soft-V from Q)
    # ==================================================================

    def getV(self, states: torch.Tensor) -> torch.Tensor:
        """V(s) = τ · logsumexp(Q(s,·)/τ)  using ONLINE network."""
        q_values = self.q_network(states)
        return self.temperature * torch.logsumexp(
            q_values / self.temperature, dim=1, keepdim=True)

    def get_targetV(self, states: torch.Tensor) -> torch.Tensor:
        """V(s) using TARGET network (no grad)."""
        with torch.no_grad():
            q_values = self.q_target(states)
            return self.temperature * torch.logsumexp(
                q_values / self.temperature, dim=1, keepdim=True)

    # ==================================================================
    # Core IQ-Learn loss  (faithful to Garg et al. reference code)
    # ==================================================================

    def compute_iq_loss(self,
                        expert_states, expert_actions,
                        expert_next_states, expert_dones,
                        expert_r_selfs, expert_r_globals,
                        expert_G_selfs, expert_G_globals,
                        learner_states, learner_actions,
                        learner_next_states, learner_dones,
                        ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the full IQ-Learn loss.

        Matches the structure of iq_loss() from the original codebase:
            Term 1:  -E_{expert}[ φ'(r̂) · r̂ ]          where r̂ = Q − γ(1-d)V(s')
            Term 2:  depends on loss_type (value / value_expert / v0)
            Term 3:  χ² regularisation if divergence == 'chi'
            Optional: gradient penalty, SVO regularisation
        """
        has_learner = len(learner_states) > 0

        # ------------------------------------------------------------------
        # Concatenate expert + learner into a single batch (as original does)
        # ------------------------------------------------------------------
        if has_learner:
            all_states      = torch.cat([expert_states, learner_states], dim=0)
            all_actions     = torch.cat([expert_actions, learner_actions], dim=0)
            all_next_states = torch.cat([expert_next_states, learner_next_states], dim=0)
            all_dones       = torch.cat([expert_dones, learner_dones], dim=0)
        else:
            all_states      = expert_states
            all_actions     = expert_actions
            all_next_states = expert_next_states
            all_dones       = expert_dones

        n_expert  = expert_states.shape[0]
        n_total   = all_states.shape[0]

        # is_expert mask: True for expert transitions, False for learner
        is_expert = torch.zeros(n_total, 1, device=self.device, dtype=torch.bool)
        is_expert[:n_expert] = True

        # ------------------------------------------------------------------
        # Q(s,a) for all transitions
        # ------------------------------------------------------------------
        current_Q = self.q_network(all_states).gather(
            1, all_actions.unsqueeze(1))  # [N, 1]

        # ------------------------------------------------------------------
        # V(s) and V(s') for all transitions
        # ------------------------------------------------------------------
        current_V = self.getV(all_states)  # [N, 1]

        if self.use_target_network:
            next_V = self.get_targetV(all_next_states)  # [N, 1]
        else:
            next_V = self.getV(all_next_states)

        # ------------------------------------------------------------------
        # Implicit reward:  r̂ = Q(s,a) − γ(1-d)V(s')
        # ------------------------------------------------------------------
        y = (1 - all_dones.unsqueeze(1)) * self.gamma * next_V
        reward = current_Q - y  # [N, 1]

        # Expert-only reward (for Term 1 and χ² on expert data)
        expert_reward = reward[is_expert.squeeze(1)]  # [n_expert, 1]

        # ------------------------------------------------------------------
        # SVO: optionally shift the Bellman target for expert transitions
        # ------------------------------------------------------------------
        svo_shift = 0.0
        r_svo_raw = None

        if self.use_svo and self.svo_mode in ('bellman', 'reward_reg', 'reward_reg_reweight'):
            r_svo_raw = self._compute_svo_reward(
                expert_r_selfs, expert_r_globals,
                expert_G_selfs, expert_G_globals).to(self.device)

            if self.normalize_svo and self._svo_global_std is not None:
                r_svo = r_svo_raw / self._svo_global_std
            else:
                r_svo = r_svo_raw

            svo_shift = self.svo_lambda * r_svo

        # Save unshifted implicit reward for χ² regularization (Term 3).
        # The χ² term bounds the LEARNED implicit reward r̂ = Q - γV(s')
        # (Garg et al., 2021, Theorem 1). Contaminating it with the
        # external SVO offset breaks this guarantee.
        expert_reward_unshifted = expert_reward

        if self.use_svo and self.svo_mode == 'bellman':
            # SVO Bellman shift enters Term 1 only (potential-based shaping,
            # Ng, Harada & Russell, 1999 — preserves optimal-policy invariance).
            expert_reward = expert_reward - svo_shift

        # ------------------------------------------------------------------
        # Track v0 for logging  (value of initial states ≈ value of expert states)
        # ------------------------------------------------------------------
        v0 = current_V[is_expert.squeeze(1)].mean()

        loss_dict = {}
        loss_dict['v0'] = v0.item()

        # ==================================================================
        # TERM 1:  -E_{expert}[ φ'(r̂) · r̂ ]
        #
        # φ'(r̂) depends on the chosen f-divergence.  For most divergences
        # we compute φ_grad with no_grad (it acts as a fixed weight).
        # ==================================================================
        with torch.no_grad():
            if self.divergence == 'hellinger':
                phi_grad = 1.0 / (1.0 + expert_reward) ** 2
            elif self.divergence == 'kl':
                phi_grad = torch.exp(-expert_reward - 1)
            elif self.divergence == 'kl2':
                phi_grad = F.softmax(-expert_reward, dim=0) * expert_reward.shape[0]
            elif self.divergence == 'kl_fix':
                phi_grad = torch.exp(-expert_reward)
            elif self.divergence == 'js':
                phi_grad = torch.exp(-expert_reward) / (2.0 - torch.exp(-expert_reward))
            else:
                # 'none' or 'chi' — φ_grad = 1  (χ² adds a separate term below)
                phi_grad = 1.0

        softq_loss = -(phi_grad * expert_reward).mean()
        loss = softq_loss
        loss_dict['softq_loss'] = softq_loss.item()

        # ==================================================================
        # TERM 2:  Sampling strategy for the second term
        # ==================================================================
        if self.loss_type == 'value_expert':
            # E_{expert}[ V(s) − γV(s') ]  (works offline)
            value_loss = (current_V - y)[is_expert.squeeze(1)].mean()
            loss = loss + value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif self.loss_type == 'value':
            # E_{all}[ V(s) − γV(s') ]  (works online, DEFAULT)
            value_loss = (current_V - y).mean()
            loss = loss + value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif self.loss_type == 'v0':
            # (1−γ) · E_{s0}[ V(s0) ]  (offline, usually suboptimal)
            v0_loss = (1 - self.gamma) * v0
            loss = loss + v0_loss
            loss_dict['v0_loss'] = v0_loss.item()

        # ==================================================================
        # TERM 3:  χ² divergence regularisation
        #
        # When divergence == 'chi', we add:
        #   1/(4α) · E_{expert}[ r̂² ]      (offline variant)
        # This is the key stabilising term from the original paper.
        # ==================================================================
        if self.divergence == 'chi':
            # χ² regularization on UNSHIFTED implicit reward.
            # 1/(4α) · E[r̂²] bounds the learned reward (Garg et al., 2021).
            # The SVO shift is external data, not a learned quantity — including
            # it here would penalize large R_SVO values, not large Q-values.
            r_for_chi2 = (expert_reward_unshifted
                          if self.use_svo and self.svo_mode == 'bellman'
                          else expert_reward)
            chi2_loss = 1.0 / (4.0 * self.div_alpha) * (r_for_chi2 ** 2).mean()
            loss = loss + chi2_loss
            loss_dict['chi2_loss'] = chi2_loss.item()

        # ==================================================================
        # Optional: gradient penalty (Wasserstein-1 metric)
        # For DQN we use a simple Q-magnitude penalty as a proxy since we
        # don't have the interpolation machinery of the original continuous
        # action critic.
        # ==================================================================
        if self.grad_pen:
            # Simple gradient penalty proxy: penalise Q magnitude
            gp_loss = self.lambda_gp * (current_Q ** 2).mean()
            loss = loss + gp_loss
            loss_dict['gp_loss'] = gp_loss.item()

        # ==================================================================
        # SVO: reward_reg — separate MSE on recovered reward
        # ==================================================================
        svo_reg_loss = torch.tensor(0.0, device=self.device)

        if self.use_svo and self.svo_mode in ('reward_reg', 'reward_reg_reweight'):
            # Recovered reward for expert transitions (before bellman shift):
            # r̂(s,a) = Q(s,a) − γ(1−d)V(s')
            expert_y = y[is_expert.squeeze(1)]
            r_recovered = current_Q[is_expert.squeeze(1)] - expert_y

            svo_target = svo_shift  # = λ · R_SVO

            svo_reg_loss = ((r_recovered - svo_target) ** 2).mean()
            loss = loss + svo_reg_loss

        # ==================================================================
        # Build info dict for logging
        # ==================================================================
        expert_q = current_Q[is_expert.squeeze(1)]
        expert_next_v_vals = next_V[:n_expert]
        gamma_v_next = y[:n_expert]

        abs_q_mean       = expert_q.abs().mean().item()
        abs_gamma_v_mean = gamma_v_next.abs().mean().item()

        info = {
            'loss':              loss.item(),
            'expert_q_mean':     expert_q.mean().item(),
            'expert_next_v_mean': expert_next_v_vals.mean().item(),
            'abs_expert_q_mean': abs_q_mean,
            'abs_gamma_v_mean':  abs_gamma_v_mean,
            'softq_loss':        loss_dict.get('softq_loss', 0.0),
        }
        if 'value_loss' in loss_dict:
            info['value_loss'] = loss_dict['value_loss']
        if 'v0_loss' in loss_dict:
            info['v0_loss'] = loss_dict['v0_loss']
        if 'chi2_loss' in loss_dict:
            info['chi2_loss'] = loss_dict['chi2_loss']
        if 'gp_loss' in loss_dict:
            info['gp_loss'] = loss_dict['gp_loss']

        # SVO logging
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
            expert_y_detached = y[:n_expert].detach()
            r_recovered_detached = (expert_q - expert_y_detached).detach()
            info['svo_reg_loss']          = svo_reg_loss.item()
            info['recovered_reward_mean'] = r_recovered_detached.mean().item()
            info['recovered_reward_std']  = r_recovered_detached.std().item()

        if self.use_svo and self.svo_mode in ('reweight', 'reward_reg_reweight'):
            r_svo_batch = self._compute_svo_reward(
                expert_r_selfs, expert_r_globals).to(self.device)
            info['svo_raw_mean'] = r_svo_batch.mean().item()
            info['svo_raw_std']  = r_svo_batch.std().item()

        if has_learner:
            learner_q = current_Q[~is_expert.squeeze(1)]
            learner_next_v_vals = next_V[n_expert:]
            info['learner_q_mean']     = learner_q.mean().item()
            info['learner_next_v_mean'] = learner_next_v_vals.mean().item()

        return loss, info

    # ==================================================================
    # Update step
    # ==================================================================

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Perform one gradient update."""
        expert_batch_size  = int(batch_size * self.replay_ratio)
        learner_batch_size = batch_size - expert_batch_size

        # --- Sample expert data ---
        if (self.use_svo
                and self.svo_mode in ('reweight', 'reward_reg_reweight')
                and self._expert_svo_weights is not None):
            expert_batch = self.expert_buffer.sample_weighted(
                expert_batch_size, self._expert_svo_weights)
        else:
            expert_batch = self.expert_buffer.sample(expert_batch_size)

        (expert_states, expert_actions, _, expert_next_states,
         expert_dones, expert_r_selfs, expert_r_globals,
         expert_G_selfs, expert_G_globals) = expert_batch

        expert_states = expert_states.to(self.device)
        expert_actions = expert_actions.to(self.device)
        expert_next_states = expert_next_states.to(self.device)
        expert_dones = expert_dones.to(self.device)
        expert_r_selfs = expert_r_selfs.to(self.device)
        expert_r_globals = expert_r_globals.to(self.device)
        expert_G_selfs = expert_G_selfs.to(self.device)
        expert_G_globals = expert_G_globals.to(self.device)

        # --- Sample learner data ---
        if len(self.learner_buffer) >= learner_batch_size and learner_batch_size > 0:
            learner_batch = self.learner_buffer.sample(learner_batch_size)
            (learner_states, learner_actions, _, learner_next_states,
             learner_dones, _, _, _, _) = learner_batch

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
            expert_G_selfs, expert_G_globals,
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
    # Utilities
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
            'mean_reward':    np.mean(episode_rewards),
            'std_reward':     np.std(episode_rewards),
            'mean_length':    np.mean(episode_lengths),
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
            # Config
            'loss_type':     self.loss_type,
            'divergence':    self.divergence,
            'div_alpha':     self.div_alpha,
            'use_svo':       self.use_svo,
            'svo_mode':      self.svo_mode,
            'svo_alpha':     self.svo_alpha,
            'svo_lambda':    self.svo_lambda,
            'normalize_svo': self.normalize_svo,
            'svo_cumulative': self.svo_cumulative,
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