import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle

class BaselineWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        config = self.env.unwrapped.config
        observation_config = config.get("observation", {})
        self.neighbour_count = observation_config.get("vehicles_count", 10)


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        r_self = reward
        r_global = self._calculate_neighbourhood_reward(action)

        info['rewards/component_self'] = r_self
        info['rewards/component_global'] = r_global

        return obs, r_self, done, truncated, info

    def _calculate_neighbourhood_reward(self, action):
        """
        Calculate neighbor rewards using highway-env's ACTUAL reward implementation.
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

        total_utility = 0.0

        for vehicle in neighbours:
            # Get all side lanes for this vehicle's current position
            try:
                side_lanes = env.road.network.all_side_lanes(vehicle.lane_index)
                num_lanes = len(side_lanes)
            except (AttributeError, TypeError):
                num_lanes = 1

            # Get lane index (use target_lane_index if it's a ControlledVehicle)
            try:
                if isinstance(vehicle, ControlledVehicle):
                    lane = vehicle.target_lane_index[2]
                else:
                    lane = vehicle.lane_index[2]
            except (AttributeError, IndexError, TypeError):
                lane = 0

            # Calculate reward components

            # 1. Collision (binary flag, will be weighted by config)
            collision_reward = float(vehicle.crashed)

            # 2. Right lane reward (normalized by number of lanes)
            right_lane_reward = lane / max(num_lanes - 1, 1)

            # 3. High speed reward (using FORWARD speed, not total speed)
            forward_speed = vehicle.speed * np.cos(vehicle.heading)
            speed_range = config.get("reward_speed_range", [20, 30])
            scaled_speed = utils.lmap(forward_speed, speed_range, [0, 1])
            high_speed_reward = np.clip(scaled_speed, 0, 1)

            # 4. On-road reward (binary flag)
            on_road_reward = float(vehicle.on_road)

            # Sum weighted rewards (matching official implementation)
            reward = (
                config.get("collision_reward", 0) * collision_reward +
                config.get("right_lane_reward", 0) * right_lane_reward +
                config.get("high_speed_reward", 0) * high_speed_reward
            )

            # Apply normalization if configured
            if config.get("normalize_reward", False):
                reward = utils.lmap(
                    reward,
                    [
                        config["collision_reward"],
                        config["high_speed_reward"] + config["right_lane_reward"],
                    ],
                    [0, 1],
                )

            # Multiply by on_road_reward (this is multiplicative, not additive!)
            reward *= on_road_reward

            total_utility += reward

        return total_utility / len(neighbours)