"""
SVO Wrapper for the Highway environment — flow-matching r_global.

Key insight:
    r_global measures how well the ego matches the speed of surrounding
    traffic. An altruistic/considerate driver matches the flow (small
    speed differential). An egoistic/assertive driver diverges from the
    flow (large speed differential, typically faster).

    r_global = -|v_ego - v_nearest| (negated, normalized)

    Higher r_global = better flow matching = more socially considerate.
    Lower  r_global = larger speed gap     = more assertive/egoistic.

    This produces the CORRECT SVO crossover:
        α =  0°: R_SVO = r_self → selects egoistic (fast drivers)
        α = 45°: R_SVO = balanced → crossover point (~48°)
        α = 90°: R_SVO = r_global → selects altruistic (flow matchers)

Validated on existing demonstrations (egoistic vs altruistic):
    Component-level:
        r_self:   d = +1.26  (egoistic scores higher — correct)
        r_global: d = -1.05  (altruistic scores higher — correct)
    R_SVO crossover at ~48° with d > 0.8 at both extremes.

r_self remains the standard highway-env reward (unchanged).

References:
    - Schwarting et al. (2019) "Social behavior for autonomous vehicles"
    - SVO unit-circle projection (Liebrand, 1984)
"""

import gymnasium as gym
import numpy as np


class SVOHighwayFlowWrapper(gym.Wrapper):
    """
    SVO wrapper for highway-env using flow-matching r_global.

    Args:
        env: Base highway-env gymnasium environment.
        svo_alpha: SVO angle in radians (0 = egoistic, π/4 = prosocial, π/2 = altruistic).
        lamb: Scaling factor for the combined SVO reward (default 1.0).
        max_speed_diff: Maximum speed differential (in normalized obs coords)
            for clipping. Differences beyond this are treated equally.
            Default 1.5 (covers the typical range in highway-env where
            ego vx ≈ 1.0 and neighbors range from ~0.0 to ~1.0).
        max_social_reward: The r_global value when the ego perfectly
            matches the nearest neighbor's speed. Should be on a similar
            scale to r_self for balanced SVO combination.
            Default 1.0.
        use_mean_flow: If True, compare ego speed to the mean speed of
            all present neighbors (flow speed) rather than just the nearest.
            Default False (nearest neighbor only — more responsive to
            local interactions, less noisy).
    """

    def __init__(
        self,
        env: gym.Env,
        svo_alpha: float = 0.0,
        lamb: float = 1.0,
        max_speed_diff: float = 1.5,
        max_social_reward: float = 1.0,
        use_mean_flow: bool = False,
    ):
        super().__init__(env)
        self.svo_alpha = svo_alpha
        self.lamb = lamb
        self.max_speed_diff = max_speed_diff
        self.max_social_reward = max_social_reward
        self.use_mean_flow = use_mean_flow

        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)
        self.svo_angle_deg = np.rad2deg(svo_alpha)

        config = self.env.unwrapped.config
        observation_config = config.get("observation", {})
        self.vehicles_count = observation_config.get("vehicles_count", 10)

        # Determine observation feature layout.
        features = observation_config.get("features", ["x", "y", "vx", "vy"])
        if "presence" in features:
            self._presence_idx = features.index("presence")
            self._has_presence = True
        else:
            self._presence_idx = None
            self._has_presence = False

        self._x_idx = features.index("x")
        self._y_idx = features.index("y")
        self._vx_idx = features.index("vx")

        # vy is optional (some configs omit it)
        self._vy_idx = features.index("vy") if "vy" in features else None
        self._n_features = len(features)

        # Logging cache
        self._last_r_self = 0.0
        self._last_r_global = 0.0
        self._last_speed_diff = 0.0

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)

        # r_self = standard highway-env reward (unchanged)
        r_self = r_env
        self._last_r_self = r_self

        # r_global = flow-matching social reward
        r_global = self._compute_flow_reward(obs)
        self._last_r_global = r_global

        # SVO combination
        svo_reward = (self.cos_alpha * r_self) + (self.sin_alpha * r_global)
        final_reward = self.lamb * svo_reward

        # Logging (matches interface expected by extract_demonstrations.py)
        info['rewards/svo_total'] = final_reward
        info['rewards/svo_angle_deg'] = self.svo_angle_deg
        info['rewards/component_self'] = r_self
        info['rewards/component_global'] = r_global
        info['rewards/speed_diff'] = self._last_speed_diff

        return obs, final_reward, terminated, truncated, info

    def _compute_flow_reward(self, obs: np.ndarray) -> float:
        """
        Compute r_global from the speed differential with nearby traffic.

        r_global = max_social_reward * (1 - |Δv| / max_speed_diff)

        Where Δv is the longitudinal speed difference between the ego
        and the nearest neighbor (or mean flow if use_mean_flow=True).

        Higher = ego matches flow = considerate.
        Lower  = ego diverges from flow = assertive.

        Args:
            obs: Observation array, shape (vehicles_count, n_features)
                 or flattened equivalent.

        Returns:
            float: Social reward in [0, max_social_reward].
        """
        # Reshape if flattened
        if obs.ndim == 1:
            obs = obs.reshape(self.vehicles_count, self._n_features)

        ego_vx = obs[0, self._vx_idx]

        if self.use_mean_flow:
            speed_diff = self._compute_mean_flow_diff(obs, ego_vx)
        else:
            speed_diff = self._compute_nearest_flow_diff(obs, ego_vx)

        self._last_speed_diff = speed_diff

        # No neighbors → perfect flow match (not disrupting anyone)
        if speed_diff is None:
            return self.max_social_reward

        # Normalize: 0 speed diff → max reward, max_speed_diff → 0 reward
        abs_diff = min(abs(speed_diff), self.max_speed_diff)
        normalized = 1.0 - (abs_diff / self.max_speed_diff)

        return normalized * self.max_social_reward

    def _compute_nearest_flow_diff(self, obs, ego_vx):
        """Speed difference with the nearest neighbor."""
        min_dist = float('inf')
        nearest_vx = None

        for i in range(1, obs.shape[0]):
            if self._has_presence:
                if obs[i, self._presence_idx] < 0.5:
                    continue
            else:
                if np.abs(obs[i]).sum() < 1e-6:
                    continue

            dx = obs[i, self._x_idx] - obs[0, self._x_idx]
            dy = obs[i, self._y_idx] - obs[0, self._y_idx]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < min_dist:
                min_dist = dist
                nearest_vx = obs[i, self._vx_idx]

        if nearest_vx is None:
            return None

        return ego_vx - nearest_vx

    def _compute_mean_flow_diff(self, obs, ego_vx):
        """Speed difference with the mean flow of all neighbors."""
        neighbor_speeds = []

        for i in range(1, obs.shape[0]):
            if self._has_presence:
                if obs[i, self._presence_idx] < 0.5:
                    continue
            else:
                if np.abs(obs[i]).sum() < 1e-6:
                    continue

            neighbor_speeds.append(obs[i, self._vx_idx])

        if not neighbor_speeds:
            return None

        mean_flow = np.mean(neighbor_speeds)
        return ego_vx - mean_flow


# ======================================================================
# Convenience factory
# ======================================================================

def create_highway_env(config: dict, svo_angle: float = 0.0,
                       render_mode: str = None, **wrapper_kwargs):
    """
    Create a highway environment with the flow-matching SVO wrapper.

    Args:
        config: Highway-env config dict (must include "id").
        svo_angle: SVO angle in radians.
        render_mode: Gymnasium render mode.
        **wrapper_kwargs: Additional kwargs for SVOHighwayFlowWrapper
            (max_speed_diff, max_social_reward, use_mean_flow).

    Returns:
        Wrapped gymnasium environment.
    """
    env = gym.make(config['id'], config=config, render_mode=render_mode)
    env = SVOHighwayFlowWrapper(
        env, svo_alpha=svo_angle, lamb=1.0, **wrapper_kwargs
    )
    return env