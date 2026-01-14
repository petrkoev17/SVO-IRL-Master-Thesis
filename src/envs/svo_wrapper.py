import gymnasium as gym
import numpy as np

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

        # Precompute angles to optimize
        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)
        self.svo_angle_deg = np.rad2deg(svo_alpha)

        # Cache config and weights
        config = self.env.unwrapped.config

        self.w_c = config["collision_reward"]
        self.w_rl = config["right_lane_reward"]
        self.w_e = config["high_speed_reward"]
        self.r_on_road = config["on_road_reward"]
        self.r_off_road = config["off_road_penalty"]

        # Speed reward range
        v_min, v_max = config["reward_speed_range"]
        self.v_min = v_min
        self.v_range_inv = 1.0 / (v_max - v_min) if v_max > v_min else 0.0

        # Neighbour count
        observation_config = config.get("observation", {})
        self.neighbour_count = observation_config.get("vehicles_count", 10)

        # Cache for logging
        self._last_r_self = 0.0
        self._last_r_global = 0.0

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        svo_reward = self._calculate_svo_reward()
        final_reward = self.lamb * svo_reward

        # Log
        info['rewards/svo_total'] = final_reward
        info['rewards/svo_angle_deg'] = self.svo_angle_deg
        info['rewards/component_self'] = self._last_r_self
        info['rewards/component_global'] = self._last_r_global

        return obs, final_reward, terminated, truncated, info

    def _calculate_svo_reward(self):
        """
        Calculate the SVO-Regularized reward.
        :return: R_svo = cos(alpha) * R_self + sin(alpha) * R_global
        """

        vehicle = self.env.unwrapped.vehicle

        # Calculate R_self
        r_self = self._calculate_single_vehicle_reward(vehicle)
        self._last_r_self = r_self

        # Egoistic optimization
        if abs(self.sin_alpha) < 1e-6:
            self._last_r_global = 0.0
            return self.cos_alpha * r_self

        # Calculate R_global
        r_global = self._calculate_neighbourhood_reward()
        self._last_r_global = r_global

        # Calculate SVO reward. Note! numpy trig functions expect radians
        #
        r_svo = self.cos_alpha * r_self + self.sin_alpha * r_global

        return r_svo

    def _calculate_single_vehicle_reward(self, vehicle):
        """
        Calculate R_self for a single vehicle.
        Formula: R_t = r_or + (w_c * r_c + w_rl * r_rl + w_e * r_e)
        :param vehicle: the vehicle to calculate the reward for
        :return: R_self
        """

        # Collision reward
        term_collision = self.w_c if vehicle.crashed else 0.0

        # Right lane reward
        lanes = vehicle.road.network.all_side_lanes(vehicle.lane_index)
        num_lanes = len(lanes)
        if num_lanes > 1:
            lane_position = vehicle.lane_index[2]
            term_right_lane = self.w_rl * (lane_position / (num_lanes - 1))
        else:
            term_right_lane = 0.0

        # High speed reward
        scaled_speed = (vehicle.speed - self.v_min) * self.v_range_inv
        r_e = max(0.0, min(1.0, scaled_speed))
        term_high_speed = self.w_e * r_e

        # On/Off road reward
        term_on_road = self.r_on_road if vehicle.on_road else self.r_off_road


        # Total reward
        return term_on_road * (term_collision + term_right_lane + term_high_speed)

    def _calculate_neighbourhood_reward(self):
        """
        Calculate R_global as the average R_self of neighbouring vehicles.
        :return: R_global
        """

        env = self.env.unwrapped

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

        total_utility = sum(
            self._calculate_single_vehicle_reward(v) for v in neighbours
        )

        return total_utility / len(neighbours)
