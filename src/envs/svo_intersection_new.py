"""
SVO Wrapper for the Intersection environment — v2 (progress-based).

Key change from v1:
    The wrapper now computes its OWN reward instead of using highway-env's
    built-in reward. This gives us a continuous progress signal that makes
    "drive carefully" optimal for prosocial agents, rather than "don't drive."

Reward structure (same for ego and neighbors):
    r = w_progress * (Δ_longitudinal / exit_distance)   # progress toward exit
      + w_speed    * clip(speed / v_max, 0, 1)           # speed incentive
      + w_collision * crashed                             # crash penalty
      + w_arrived  * arrived                              # terminal bonus

    For the ego vehicle, r_self uses this formula.
    For neighbors, r_global averages (or min-pools) this formula across
    all nearby vehicles. This means a cooperative agent benefits from
    neighbors making progress, not just from neighbors not crashing.

Progress tracking:
    Each step, we compute the vehicle's longitudinal position along its
    current lane using lane.local_coordinates(position)[0]. The delta
    between current and previous position gives the per-step progress.
    This is normalised by exit_distance (default 25m) to give a [0,1]-ish
    scale per step.

    Lane transitions are handled explicitly: when a vehicle moves from one
    lane segment to the next (e.g., approach → internal → exit), the
    longitudinal coordinate resets to 0. We detect this by tracking the
    lane_index alongside the longitudinal position and awarding a segment
    transition bonus equal to the remaining distance in the old segment.

References:
    - Schwarting et al. (2019) "Social behavior for autonomous vehicles" PNAS
    - Tong et al. (2024) "Human-like decision making at unsignalized
      intersections using SVO" IEEE ITS Magazine
    - Hierarchical reward literature (see intersection_config.py docstring)
"""

import gymnasium as gym
import numpy as np
from highway_env import utils


class SVOIntersectionWrapper(gym.Wrapper):
    """
    SVO Wrapper for intersection with progress-based reward.

    global_aggregation controls how neighbor rewards are combined:
        'mean': Average of all neighbor rewards.
        'min': Minimum neighbor reward — captures worst-affected vehicle.
    """

    def __init__(self, env: gym.Env, svo_alpha: float = 0.0, lamb: float = 1.0,
                 global_aggregation: str = 'mean'):
        super().__init__(env)
        self.svo_alpha = svo_alpha
        self.lamb = lamb
        self.global_aggregation = global_aggregation

        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)
        self.svo_angle_deg = np.rad2deg(svo_alpha)

        config = self.env.unwrapped.config
        observation_config = config.get("observation", {})
        self.neighbour_count = observation_config.get("vehicles_count", 10)

        # Reward weights from config
        self.w_progress = config.get("progress_reward", 2.0)
        self.w_speed = config.get("high_speed_reward", 0.5)
        self.w_collision = config.get("collision_reward", -2.0)
        self.w_arrived = config.get("arrived_reward", 1.0)
        self.w_timeout = config.get("timeout_reward", -0.5)
        self.speed_range = config.get("reward_speed_range", [7.0, 9.0])
        self.exit_distance = config.get("exit_distance", 25.0)
        self.neighbour_distance = config.get("neighbour_distance", 30.0)

        # Progress tracking — ego: (lane_index, longitudinal)
        self._ego_prev_state = None

        # Progress tracking — neighbors: vid -> (lane_index, longitudinal)
        self._neighbor_prev_state = {}

        # Logging cache
        self._last_r_self = 0.0
        self._last_r_global = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialise ego progress tracking
        ego = self.env.unwrapped.vehicle
        self._ego_prev_state = self._get_progress_state(ego)

        # Initialise neighbor tracking
        self._neighbor_prev_state = {}
        for v in self.env.unwrapped.road.vehicles:
            if v is not ego:
                self._neighbor_prev_state[id(v)] = self._get_progress_state(v)

        return obs, info

    def step(self, action):
        obs, _r_env, terminated, truncated, info = self.env.step(action)

        env = self.env.unwrapped
        ego = env.vehicle

        # ----- Compute r_self (ego reward) -----
        r_self = self._compute_vehicle_reward(ego, self._ego_prev_state)

        # Add timeout penalty if episode ends without arrival or crash
        if truncated and not terminated and not self._has_arrived(ego):
            r_self += self.w_timeout

        # Update ego tracking
        self._ego_prev_state = self._get_progress_state(ego)

        self._last_r_self = r_self

        # ----- Compute r_global (neighbor rewards) -----
        r_global = self._calculate_neighbourhood_reward()
        self._last_r_global = r_global

        # ----- SVO combination -----
        svo_reward = (self.cos_alpha * r_self) + (self.sin_alpha * r_global)
        final_reward = self.lamb * svo_reward

        # Log
        info['rewards/svo_total'] = final_reward
        info['rewards/svo_angle_deg'] = self.svo_angle_deg
        info['rewards/component_self'] = r_self
        info['rewards/component_global'] = r_global

        return obs, final_reward, terminated, truncated, info

    # ==================================================================
    # Core reward computation (shared by ego and neighbors)
    # ==================================================================

    def _compute_vehicle_reward(self, vehicle, prev_state):
        """
        Compute reward for a single vehicle using progress-based structure.

        Args:
            vehicle: The vehicle to compute reward for.
            prev_state: Previous (lane_index, longitudinal) tuple, or None.

        Returns:
            float: The reward for this vehicle this step.
        """
        # 1. Collision (binary) — overrides everything
        if vehicle.crashed:
            return self.w_collision

        # 2. Arrived (terminal bonus) — overrides progress/speed
        if self._has_arrived(vehicle):
            return self.w_arrived

        # 3. Progress reward with lane transition handling
        curr_lane_index = self._get_lane_index(vehicle)
        curr_longi = self._get_longitudinal(vehicle)

        progress = 0.0
        if prev_state is not None:
            prev_lane_index, prev_longi = prev_state

            if curr_lane_index == prev_lane_index:
                # Same lane segment — simple delta
                delta = curr_longi - prev_longi
                progress = max(0.0, delta) / self.exit_distance
            else:
                # Lane transition detected (e.g., approach -> internal -> exit)
                # The vehicle moved from prev_lane to curr_lane.
                # Credit: remaining distance in old lane + distance in new lane.
                prev_lane_remaining = self._get_lane_length(vehicle, prev_lane_index) - prev_longi
                prev_lane_remaining = max(0.0, prev_lane_remaining)
                transition_progress = prev_lane_remaining + curr_longi
                progress = max(0.0, transition_progress) / self.exit_distance

        # 4. Speed reward: clip(speed / v_max, 0, 1)
        scaled_speed = utils.lmap(vehicle.speed, self.speed_range, [0, 1])
        speed_reward = np.clip(scaled_speed, 0, 1)

        # 5. On-road check (multiplicative — zero reward if off-road)
        on_road = float(vehicle.on_road)

        reward = (
            self.w_progress * progress +
            self.w_speed * speed_reward
        ) * on_road

        return reward

    # ==================================================================
    # Neighbor reward
    # ==================================================================

    def _calculate_neighbourhood_reward(self):
        """
        Compute progress-based reward for all neighbors, then aggregate.
        """
        env = self.env.unwrapped
        ego = env.vehicle

        neighbours = env.road.close_vehicles_to(
            vehicle=ego,
            distance=self.neighbour_distance,
            count=self.neighbour_count,
            sort=True
        )
        neighbours = [v for v in neighbours if v is not ego]

        if not neighbours:
            return 0.0

        per_vehicle_rewards = []

        for vehicle in neighbours:
            vid = id(vehicle)
            prev_state = self._neighbor_prev_state.get(vid, None)

            reward = self._compute_vehicle_reward(vehicle, prev_state)
            per_vehicle_rewards.append(reward)

            # Update tracking
            self._neighbor_prev_state[vid] = self._get_progress_state(vehicle)

        # Clean up vehicles that are no longer in the scene
        current_vids = {id(v) for v in env.road.vehicles if v is not ego}
        stale = [vid for vid in self._neighbor_prev_state if vid not in current_vids]
        for vid in stale:
            del self._neighbor_prev_state[vid]

        # Aggregate
        if self.global_aggregation == 'min':
            return min(per_vehicle_rewards)
        else:
            return sum(per_vehicle_rewards) / len(per_vehicle_rewards)

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _get_longitudinal(vehicle):
        """
        Get the longitudinal position of a vehicle along its current lane.
        Returns the local longitudinal coordinate (meters from lane start).
        """
        try:
            longi, _ = vehicle.lane.local_coordinates(vehicle.position)
            return float(longi)
        except (AttributeError, TypeError):
            return 0.0

    @staticmethod
    def _get_lane_index(vehicle):
        """Get the vehicle's current lane index as a tuple."""
        try:
            return tuple(vehicle.lane_index)
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _get_lane_length(vehicle, lane_index):
        """
        Get the length of a lane segment by its index.
        Falls back to exit_distance (25m) if lookup fails.
        """
        try:
            road_network = vehicle.road.network
            lane = road_network.get_lane(lane_index)
            return lane.length
        except (AttributeError, TypeError, KeyError):
            return 25.0

    def _get_progress_state(self, vehicle):
        """Get the (lane_index, longitudinal) state for progress tracking."""
        return (self._get_lane_index(vehicle), self._get_longitudinal(vehicle))

    def _has_arrived(self, vehicle):
        """Check if a vehicle has reached an intersection exit."""
        try:
            return (
                "il" in vehicle.lane_index[0]
                and "o" in vehicle.lane_index[1]
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= self.exit_distance
            )
        except (AttributeError, TypeError, IndexError):
            return False