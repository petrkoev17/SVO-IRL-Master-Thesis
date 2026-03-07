import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle


class SVOIntersectionWrapper(gym.Wrapper):
    """
    SVO Wrapper adapted for the Intersection environment.

    Key differences from SVOPureWrapper (highway):
        - No right_lane_reward (intersections don't have parallel lanes)
        - Has arrived_reward (reaching destination overrides other rewards)
        - Speed range is [7, 9] m/s instead of [20, 30] m/s
        - Collision reward is -5 (not -1)

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

        # Logging cache
        self._last_r_self = 0.0
        self._last_r_global = 0.0

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)

        r_self = r_env
        self._last_r_self = r_self

        r_global = self._calculate_neighbourhood_reward(action)
        self._last_r_global = r_global

        svo_reward = (self.cos_alpha * r_self) + (self.sin_alpha * r_global)
        final_reward = self.lamb * svo_reward

        info['rewards/svo_total'] = final_reward
        info['rewards/svo_angle_deg'] = self.svo_angle_deg
        info['rewards/component_self'] = r_self
        info['rewards/component_global'] = r_global

        return obs, final_reward, terminated, truncated, info

    def _calculate_neighbourhood_reward(self, action):
        """
        Calculate neighbor rewards using the intersection reward structure.

        Matches IntersectionEnv._agent_rewards:
            collision_reward * crashed
            high_speed_reward * clip(lmap(speed, speed_range, [0,1]), 0, 1)
            arrived_reward (overrides if vehicle has arrived)
            on_road_reward (multiplicative)
        """
        env = self.env.unwrapped
        original_ego = env.vehicle
        config = env.config

        neighbours = env.road.close_vehicles_to(
            vehicle=original_ego,
            distance=50.0,
            count=self.neighbour_count,
            sort=True
        )
        neighbours = [v for v in neighbours if v is not original_ego]

        if not neighbours:
            return 0.0

        per_vehicle_rewards = []

        for vehicle in neighbours:
            # 1. Collision
            collision_reward = float(vehicle.crashed)

            # 2. High speed reward (using total speed, not just forward component)
            # Intersection vehicles move in multiple directions
            speed_range = config.get("reward_speed_range", [7.0, 9.0])
            scaled_speed = utils.lmap(vehicle.speed, speed_range, [0, 1])
            high_speed_reward = np.clip(scaled_speed, 0, 1)

            # 3. Arrived reward — check if this neighbor has reached an exit
            has_arrived = (
                "il" in vehicle.lane_index[0]
                and "o" in vehicle.lane_index[1]
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= 25
            )

            # 4. On-road reward
            on_road_reward = float(vehicle.on_road)

            # Compute reward (matching _agent_reward logic)
            reward = (
                config.get("collision_reward", -5) * collision_reward +
                config.get("high_speed_reward", 1) * high_speed_reward
            )

            # arrived_reward overrides if vehicle has arrived
            if has_arrived:
                reward = config.get("arrived_reward", 1)

            # Normalize if configured
            if config.get("normalize_reward", False):
                reward = utils.lmap(
                    reward,
                    [config["collision_reward"], config["arrived_reward"]],
                    [0, 1],
                )

            # Multiply by on_road_reward (multiplicative)
            reward *= on_road_reward

            per_vehicle_rewards.append(reward)

        # Aggregate
        if self.global_aggregation == 'min':
            return min(per_vehicle_rewards)
        else:
            return sum(per_vehicle_rewards) / len(per_vehicle_rewards)