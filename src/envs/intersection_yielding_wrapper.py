"""
SVO Wrapper for intersection-v1 — Counterfactual Yielding r_global.

Transfers the gridworld v2 reward decomposition to continuous space:

r_self (every step):
    Same as SVOIntersectionWrapper v2:
    progress + speed + collision + arrival + timeout

r_global (counterfactual yielding):
    Only nonzero when ego is in the "influence zone" — approaching the
    intersection conflict area with cross-traffic nearby.

    Positive: ego decelerates/stops while neighbors approach the conflict
              zone (yielding — ego sacrifices speed for others' safety)
    Negative: ego maintains speed and forces neighbors to brake
              (disruption — ego's action costs others)
    Zero:     ego is far from the conflict zone, or no cross-traffic nearby

    This prevents r_global farming: ego can't collect social reward
    by sitting passively where it doesn't affect anyone.

Influence zone detection:
    Ego is "approaching conflict" when on an approach lane segment
    (lane_index[0] starts with 'o', lane_index[1] starts with 'il')
    and longitudinal position > approach_threshold (near the end of
    the approach lane, about to enter the intersection).

    Ego is "in conflict" when on an internal lane segment
    ('il' in both lane_index[0] and lane_index[1]).

Cross-traffic detection:
    Neighbors on different approach arms within conflict_distance meters
    of ego, who are heading toward the same conflict area.
"""

import gymnasium as gym
import numpy as np
from highway_env import utils


class SVOYieldingWrapper(gym.Wrapper):
    """
    SVO Wrapper with counterfactual yielding r_global.
    Drop-in replacement for SVOIntersectionWrapper.
    """

    def __init__(self, env: gym.Env, svo_alpha: float = 0.0, lamb: float = 1.0,
                 global_aggregation: str = 'mean'):
        super().__init__(env)
        self.svo_alpha = svo_alpha
        self.lamb = lamb

        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)
        self.svo_angle_deg = np.rad2deg(svo_alpha)

        config = self.env.unwrapped.config
        obs_config = config.get("observation", {})
        self.neighbour_count = obs_config.get("vehicles_count", 10)

        # r_self weights (same as existing wrapper)
        self.w_progress = config.get("progress_reward", 2.0)
        self.w_speed = config.get("high_speed_reward", 0.5)
        self.w_collision = config.get("collision_reward", -2.0)
        self.w_arrived = config.get("arrived_reward", 1.0)
        self.w_timeout = config.get("timeout_reward", -0.5)
        self.speed_range = config.get("reward_speed_range", [7.0, 9.0])
        self.exit_distance = config.get("exit_distance", 25.0)
        self.w_step = config.get("step_cost", -0.15)

        # r_global weights (counterfactual yielding)
        self.w_yield = config.get("yield_reward", 0.5)
        self.w_disruption = config.get("disruption_penalty", -1.5)
        self.w_other_collision = config.get("other_collision_penalty", -5.0)

        # Influence zone parameters
        self.neighbour_distance = config.get("neighbour_distance", 30.0)
        self.approach_threshold = config.get("approach_threshold", 8.0)
        # How close a neighbor must be (meters) to count as "approaching conflict"
        self.conflict_distance = config.get("conflict_distance", 20.0)

        # Ego progress tracking
        self._ego_prev_state = None
        # Neighbor speed tracking (for disruption measurement)
        self._prev_neighbor_speeds = {}

        # Logging
        self._last_r_self = 0.0
        self._last_r_global = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        ego = self.env.unwrapped.vehicle
        self._ego_prev_state = self._get_progress_state(ego)
        self._prev_neighbor_speeds = {}

        for v in self.env.unwrapped.road.vehicles:
            if v is not ego:
                self._prev_neighbor_speeds[id(v)] = v.speed

        return obs, info

    def step(self, action):
        # Capture pre-step state
        env = self.env.unwrapped
        ego = env.vehicle
        prev_ego_speed = ego.speed
        prev_ego_lane = self._get_lane_index(ego)
        prev_ego_longi = self._get_longitudinal(ego)

        # Snapshot neighbor speeds before step
        pre_step_neighbor_speeds = {}
        for v in env.road.vehicles:
            if v is not ego:
                pre_step_neighbor_speeds[id(v)] = v.speed

        # Step the environment
        obs, _r_env, terminated, truncated, info = self.env.step(action)

        ego = env.vehicle  # re-fetch in case reference changed

        # ── r_self ──
        r_self = self._compute_r_self(ego, prev_ego_lane, prev_ego_longi,
                                       terminated, truncated)

        # ── r_global ──
        r_global = self._compute_r_global(
            ego, prev_ego_speed, prev_ego_lane, prev_ego_longi,
            pre_step_neighbor_speeds
        )

        # Update tracking
        self._ego_prev_state = self._get_progress_state(ego)
        self._prev_neighbor_speeds = {}
        for v in env.road.vehicles:
            if v is not ego:
                self._prev_neighbor_speeds[id(v)] = v.speed

        self._last_r_self = r_self
        self._last_r_global = r_global

        # SVO combination
        svo_reward = self.cos_alpha * r_self + self.sin_alpha * r_global
        final_reward = self.lamb * svo_reward

        info['rewards/svo_total'] = final_reward
        info['rewards/svo_angle_deg'] = self.svo_angle_deg
        info['rewards/component_self'] = r_self
        info['rewards/component_global'] = r_global

        return obs, final_reward, terminated, truncated, info

    # ==================================================================
    # r_self — same structure as existing wrapper
    # ==================================================================

    def _compute_r_self(self, ego, prev_lane, prev_longi, terminated, truncated):
        """Progress-based ego reward."""
        if ego.crashed:
            return self.w_collision

        if self._has_arrived(ego):
            return self.w_arrived

        # Progress
        curr_lane = self._get_lane_index(ego)
        curr_longi = self._get_longitudinal(ego)
        progress = 0.0

        if prev_lane is not None:
            if curr_lane == prev_lane:
                delta = curr_longi - prev_longi
                progress = max(0.0, delta) / self.exit_distance
            else:
                prev_remaining = self._get_lane_length(ego, prev_lane) - prev_longi
                prev_remaining = max(0.0, prev_remaining)
                progress = max(0.0, prev_remaining + curr_longi) / self.exit_distance

        # Speed
        scaled_speed = utils.lmap(ego.speed, self.speed_range, [0, 1])
        speed_reward = np.clip(scaled_speed, 0, 1)

        on_road = float(ego.on_road)
        r = (self.w_progress * progress + self.w_speed * speed_reward) * on_road
        r += self.w_step

        # Timeout
        if truncated and not terminated and not self._has_arrived(ego):
            r += self.w_timeout

        return r

    # ==================================================================
    # r_global — counterfactual yielding
    # ==================================================================

    def _compute_r_global(self, ego, prev_ego_speed, prev_ego_lane,
                          prev_ego_longi, pre_step_neighbor_speeds):
        env = self.env.unwrapped

        # Neighbor collision check
        for v in env.road.vehicles:
            if v is not ego and v.crashed:
                vid = id(v)
                if vid in pre_step_neighbor_speeds:
                    return self.w_other_collision

        # Find all nearby neighbors
        neighbors = env.road.close_vehicles_to(
            vehicle=ego, distance=self.conflict_distance,
            count=self.neighbour_count, sort=True
        )
        neighbors = [v for v in neighbors if v is not ego]

        if not neighbors:
            return 0.0

        n = len(neighbors)
        r_global = 0.0

        # Yielding: ego going slow relative to max while neighbors are near
        speed_ratio = ego.speed / max(self.speed_range[1], 1.0)
        yield_signal = (1.0 - speed_ratio)  # 1.0 when stopped, 0.0 at max speed
        r_global += self.w_yield * yield_signal * n

        # Disruption: neighbors braked because of ego
        for v in neighbors:
            vid = id(v)
            prev_speed = pre_step_neighbor_speeds.get(vid, v.speed)
            speed_loss = prev_speed - v.speed
            if speed_loss > 1.0:
                disruption_frac = min(speed_loss / 5.0, 1.0)
                r_global += self.w_disruption * disruption_frac

        return r_global

    # ==================================================================
    # Influence zone detection
    # ==================================================================

    def _is_approaching_conflict(self, lane_index, longi):
        """
        Is ego on an approach lane, near the end (about to enter intersection)?
        Approach lanes: lane_index[0] starts with 'o', lane_index[1] starts with 'il'
        """
        if lane_index is None:
            return False
        try:
            origin, dest = lane_index[0], lane_index[1]
            is_approach = (origin.startswith('o') and dest.startswith('il'))
            near_end = longi >= self.approach_threshold
            return is_approach and near_end
        except (IndexError, AttributeError):
            return False

    def _is_in_conflict(self, lane_index):
        """
        Is ego on an internal intersection lane?
        Internal lanes: both lane_index[0] and [1] contain 'il'
        """
        if lane_index is None:
            return False
        try:
            return 'il' in lane_index[0] and 'il' in lane_index[1]
        except (IndexError, AttributeError):
            return False

    def _get_approaching_cross_traffic(self, ego):
        """
        Find neighbor vehicles that are:
        1. On a different approach arm than ego
        2. Within conflict_distance meters
        3. Heading toward the conflict zone (not already past it)
        """
        env = self.env.unwrapped
        ego_lane = self._get_lane_index(ego)

        if ego_lane is None:
            return []

        # Ego's approach arm (e.g., 'o0', 'o1', 'o2', 'o3')
        ego_origin = ego_lane[0] if ego_lane else None

        neighbors = env.road.close_vehicles_to(
            vehicle=ego,
            distance=self.conflict_distance,
            count=self.neighbour_count,
            sort=True,
        )

        approaching = []
        for v in neighbors:
            if v is ego:
                continue
            v_lane = self._get_lane_index(v)
            if v_lane is None:
                continue

            v_origin = v_lane[0]

            # Different approach arm (cross traffic, not same direction)
            if v_origin == ego_origin:
                continue

            # Is the neighbor approaching or in the conflict zone?
            v_longi = self._get_longitudinal(v)
            v_approaching = (v_origin.startswith('o') and v_lane[1].startswith('il'))
            v_in_conflict = ('il' in v_origin and 'il' in v_lane[1])

            if v_approaching or v_in_conflict:
                approaching.append(v)

        return approaching

    # ==================================================================
    # Helpers (same as existing wrapper)
    # ==================================================================

    @staticmethod
    def _get_longitudinal(vehicle):
        try:
            longi, _ = vehicle.lane.local_coordinates(vehicle.position)
            return float(longi)
        except (AttributeError, TypeError):
            return 0.0

    @staticmethod
    def _get_lane_index(vehicle):
        try:
            return tuple(vehicle.lane_index)
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _get_lane_length(vehicle, lane_index):
        try:
            lane = vehicle.road.network.get_lane(lane_index)
            return lane.length
        except (AttributeError, TypeError, KeyError):
            return 25.0

    def _get_progress_state(self, vehicle):
        return (self._get_lane_index(vehicle), self._get_longitudinal(vehicle))

    def _has_arrived(self, vehicle):
        try:
            return (
                "il" in vehicle.lane_index[0]
                and "o" in vehicle.lane_index[1]
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= self.exit_distance
            )
        except (AttributeError, TypeError, IndexError):
            return False