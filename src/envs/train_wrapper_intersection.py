"""
SVO Wrapper for intersection-v1 — Training version.

Purpose: Train new expert policies with proper conflict-zone yielding.
         This wrapper should NOT be used to evaluate old experts trained
         with the progress-based wrapper.

r_self (every step):
    progress + speed + collision + arrival + step_cost + timeout

r_global (conflict-zone yielding — matches gridworld v2):
    YIELD REWARD: ego is on approach lane near the intersection AND
                  decelerates/stops while cross-traffic approaches.
                  Positive, proportional to number of approaching vehicles.

    DISRUPTION PENALTY: ego is on or crossing through the internal lanes
                        AND forces cross-traffic to brake.
                        Negative, proportional to speed loss caused.

    ZERO: ego is far from the conflict zone, past the intersection,
          or no cross-traffic nearby.

Key difference from svo_yielding_wrapper.py (evaluation version):
    - Uses lane geometry to detect the conflict zone precisely
    - Only rewards yielding at the approach lane (not "being slow anywhere")
    - Cross-traffic must be on a different approach arm
    - Matches the gridworld v2 logic: yield at x=2 (approach), disruption at x=3 (conflict)
"""

import gymnasium as gym
import numpy as np
from highway_env import utils


class SVOYieldingTrainWrapper(gym.Wrapper):
    """
    SVO Wrapper for training experts with conflict-zone yielding r_global.
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

        # r_self weights
        self.w_progress = config.get("progress_reward", 2.0)
        self.w_speed = config.get("high_speed_reward", 0.5)
        self.w_collision = config.get("collision_reward", -2.0)
        self.w_arrived = config.get("arrived_reward", 1.0)
        self.w_timeout = config.get("timeout_reward", -0.5)
        self.w_step = config.get("step_cost", -0.15)
        self.speed_range = config.get("reward_speed_range", [7.0, 9.0])
        self.exit_distance = config.get("exit_distance", 25.0)

        # r_global weights
        self.w_yield = config.get("yield_reward", 0.5)
        self.w_disruption = config.get("disruption_penalty", -1.5)
        self.w_other_collision = config.get("other_collision_penalty", -5.0)

        # Conflict zone geometry
        self.neighbour_distance = config.get("neighbour_distance", 30.0)
        # Ego must be past this longitudinal position on approach lane
        # to be at the "yield decision point" (analogous to gridworld x=2)
        self.approach_threshold = config.get("approach_threshold", 8.0)
        # Max distance to search for cross-traffic
        self.conflict_distance = config.get("conflict_distance", 25.0)

        # State tracking
        self._ego_prev_state = None
        self._prev_neighbor_speeds = {}

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
        env = self.env.unwrapped
        ego = env.vehicle

        # Capture pre-step state
        prev_ego_speed = ego.speed
        prev_ego_lane = self._get_lane_index(ego)
        prev_ego_longi = self._get_longitudinal(ego)

        pre_step_neighbor_speeds = {}
        for v in env.road.vehicles:
            if v is not ego:
                pre_step_neighbor_speeds[id(v)] = v.speed

        # Step
        obs, _r_env, terminated, truncated, info = self.env.step(action)
        ego = env.vehicle

        # Compute rewards
        r_self = self._compute_r_self(ego, prev_ego_lane, prev_ego_longi,
                                       terminated, truncated)

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
    # r_self
    # ==================================================================

    def _compute_r_self(self, ego, prev_lane, prev_longi, terminated, truncated):
        if ego.crashed:
            return self.w_collision

        if self._has_arrived(ego):
            return self.w_arrived

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

        scaled_speed = utils.lmap(ego.speed, self.speed_range, [0, 1])
        speed_reward = np.clip(scaled_speed, 0, 1)

        on_road = float(ego.on_road)
        r = (self.w_progress * progress + self.w_speed * speed_reward) * on_road
        r += self.w_step

        if truncated and not terminated and not self._has_arrived(ego):
            r += self.w_timeout

        return r

    # ==================================================================
    # r_global — conflict-zone yielding (matches gridworld v2)
    # ==================================================================

    def _compute_r_global(self, ego, prev_ego_speed, prev_ego_lane,
                           prev_ego_longi, pre_step_neighbor_speeds):
        """
        Conflict-zone aware r_global.

        Analogous to gridworld v2:
        - Approach lane (near end) = gridworld x=2 → yield reward
        - Internal lane = gridworld x=3 → disruption penalty
        - Everywhere else = 0
        """
        env = self.env.unwrapped

        # Neighbor collision
        for v in env.road.vehicles:
            if v is not ego and v.crashed:
                if id(v) in pre_step_neighbor_speeds:
                    return self.w_other_collision

        # Determine ego's zone from PRE-STEP position
        ego_approaching = self._is_approaching_conflict(prev_ego_lane, prev_ego_longi)
        ego_in_conflict = self._is_in_conflict(prev_ego_lane)

        if not ego_approaching and not ego_in_conflict:
            return 0.0

        # Find cross-traffic (different approach arm, heading toward conflict)
        approaching_cross = self._get_approaching_cross_traffic(ego)

        if not approaching_cross:
            return 0.0

        r_global = 0.0
        n_cross = len(approaching_cross)

        # ── Yield reward: ego on approach lane, decelerating ──
        # Analogous to gridworld: ego at x=2, action=STOP → positive r_global
        if ego_approaching:
            ego_decelerated = ego.speed < prev_ego_speed - 0.5
            ego_stopped = ego.speed < 0.5

            if ego_stopped:
                r_global += self.w_yield * n_cross
            elif ego_decelerated:
                r_global += (self.w_yield * 0.5) * n_cross

        # ── Disruption penalty: ego in/crossing conflict, neighbors brake ──
        # Analogous to gridworld: ego at x=3 or crossing from x=2 → negative r_global
        if ego_in_conflict or (ego_approaching and not (ego.speed < prev_ego_speed - 0.5)):
            for v in approaching_cross:
                vid = id(v)
                prev_speed = pre_step_neighbor_speeds.get(vid, v.speed)
                speed_loss = prev_speed - v.speed

                if speed_loss > 1.0:
                    disruption_frac = min(speed_loss / 5.0, 1.0)
                    r_global += self.w_disruption * disruption_frac

        return r_global

    # ==================================================================
    # Zone detection
    # ==================================================================

    def _is_approaching_conflict(self, lane_index, longi):
        """Ego on approach lane, near the end → yield decision point."""
        if lane_index is None:
            return False
        try:
            origin, dest = lane_index[0], lane_index[1]
            is_approach = origin.startswith('o') and dest.startswith('il')
            return is_approach and longi >= self.approach_threshold
        except (IndexError, AttributeError):
            return False

    def _is_in_conflict(self, lane_index):
        """Ego on internal intersection lane → in the conflict zone."""
        if lane_index is None:
            return False
        try:
            return 'il' in lane_index[0] and 'il' in lane_index[1]
        except (IndexError, AttributeError):
            return False

    def _get_approaching_cross_traffic(self, ego):
        """Find neighbors on different approach arms heading toward conflict."""
        env = self.env.unwrapped
        ego_lane = self._get_lane_index(ego)
        if ego_lane is None:
            return []

        ego_origin = ego_lane[0]

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

            # Must be on a different approach arm
            if v_lane[0] == ego_origin:
                continue

            # Must be approaching or in the conflict zone
            v_approaching = v_lane[0].startswith('o') and v_lane[1].startswith('il')
            v_in_conflict = 'il' in v_lane[0] and 'il' in v_lane[1]

            if v_approaching or v_in_conflict:
                approaching.append(v)

        return approaching

    # ==================================================================
    # Helpers
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