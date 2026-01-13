import gymnasium as gym
import numpy as np
from highway_env.envs import HighwayEnv

class SVOWrapper(gym.Wrapper):
    """
    Wrapper for the HighwayEnv to implement SVO-Regularized reward function.

    The total reward is:
    R_SVO = cos(alpha) * R_self + sin(alpha) * R_global
    """

    def __init__(self, env: gym.Env, svo_alpha: float = 0.0, lamb: float = 1.0):
        super().__init__(env)
        self.svo_alpha = svo_alpha
        self.lamb = lamb

        # Load configuration
        config = getattr(self.env.unwrapped, "config", {})

        # Thesis Eq (3) Weights:
        self.w_c = config.get("collision_reward", -5.0)
        self.w_rl = config.get("right_lane_reward", 0.1)
        self.w_e = config.get("high_speed_reward", 0.4)

        # On/Off road weights
        self.r_on_road = config.get("on_road_reward", 1.5)
        self.r_off_road = config.get("off_road_penalty", -10.0)

        # Speed reward range
        self.speed_range = config.get("reward_speed_range", [20.0, 30.0])
        self.v_min = self.speed_range[0]
        self.v_max = self.speed_range[1]

        # Neighbour count
        observation_config = config.get("observation", {})
        self.neighbour_count = observation_config.get("vehicles_count", 10)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        svo_reward = self._calculate_svo_reward()
        final_reward = self.lamb * svo_reward

        # Log
        info['rewards/svo_total'] = final_reward
        info['rewards/svo_angle_deg'] = np.rad2deg(self.svo_alpha)

        if hasattr(self, '_last_r_self'):
            info['rewards/component_self'] = self._last_r_self
            info['rewards/component_global'] = self._last_r_global

        return obs, final_reward, terminated, truncated, info

    def _calculate_svo_reward(self):
        """
        Calculate the SVO-Regularized reward.
        :return: R_svo = cos(alpha) * R_self + sin(alpha) * R_global
        """

        env = self.unwrapped
        vehicle = env.vehicle

        # Calculate R_self
        r_self = self._calculate_single_vehicle_reward(vehicle)
        self._last_r_self = r_self

        # Calculate R_global
        r_global = self._calculate_neighbourhood_reward(env)
        self._last_r_global = r_global

        # Calculate SVO reward. Note! numpy trig functions expect radians
        #
        r_svo = np.cos(self.svo_alpha) * r_self + np.sin(self.svo_alpha) * r_global

        return r_svo

    def _calculate_single_vehicle_reward(self, vehicle):
        """
        Calculate R_self for a single vehicle.
        Formula: R_t = r_or + (w_c * r_c + w_rl * r_rl + w_e * r_e)
        :param vehicle: the vehicle to calculate the reward for
        :return: R_self
        """

        # Collision reward
        r_c = 1.0 if vehicle.crashed else 0.0
        term_collision = self.w_c * r_c

        # Right lane reward
        lanes = vehicle.road.network.all_side_lanes(vehicle.lane_index)
        num_lanes = len(lanes)
        if num_lanes > 1:
            r_rl = vehicle.lane_index[2] / (num_lanes - 1)
        else:
            r_rl = 0.0

        term_right_lane = self.w_rl * r_rl

        # High speed reward
        scaled_speed = (vehicle.speed - self.v_min) / (self.v_max - self.v_min)
        r_e = np.clip(scaled_speed, 0.0, 1.0)
        term_high_speed = self.w_e * r_e

        # On/Off road reward
        if vehicle.on_road:
            term_on_road = self.r_on_road
        else:
            term_on_road = self.r_off_road


        # Total reward
        total_reward = term_on_road + term_collision + term_right_lane + term_high_speed
        return total_reward

    def _calculate_neighbourhood_reward(self, env):
        """
        Calculate R_global as the average R_self of neighbouring vehicles.
        :param env: Environment
        :return: R_global
        """

        # Get neighbouring vehicles
        neighbours = env.road.close_vehicles_to(
            vehicle=env.vehicle,
            distance=50.0,
            count=self.neighbour_count,
            sort = True,
        )

        neighbours = [v for v in neighbours if v is not env.vehicle]

        if not neighbours:
            return 0.0

        total_utility = 0.0
        for v in neighbours:
            total_utility += self._calculate_single_vehicle_reward(v)

        return total_utility / len(neighbours)
