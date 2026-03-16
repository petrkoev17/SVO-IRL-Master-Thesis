import numpy as np
from highway_env.vehicle.behavior import IDMVehicle


class RandomizedIDMVehicle(IDMVehicle):
    """
    A vehicle that randomly samples its driving personality from scientific distributions
    upon initialization.
    Source: Kesting, A., Treiber, M., & Helbing, D. (2008).
    'Calibrating car-following models by using trajectory data.'

    Speed range aligned with reward_speed_range: [20, 30] m/s
    """

    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None,
                 lane_index=None, route=None, **kwargs):
        # Accept and ignore extra kwargs that highway-env might pass
        super().__init__(road, position, heading, speed, target_lane_index,
                         lane_index, route)

        # Only randomize if this is a NEW vehicle (not a copy)
        # If target_speed is passed, it means we're copying an existing vehicle
        if 'target_speed' not in kwargs:
            self._randomize_parameters()
        else:
            # Use the passed parameters (vehicle is being copied)
            for key, value in kwargs.items():
                if hasattr(self, key.upper()):
                    setattr(self, key.upper(), value)
                elif hasattr(self, key):
                    setattr(self, key, value)

    # def _randomize_parameters(self):
    #     """Randomize all driving parameters"""
    #     # --- 1. Randomize Desired Speed (v0) ---
    #
    #     lane_speed_limit = self.lane.speed_limit
    #     self.target_speed = np.clip(
    #         np.random.normal(loc=lane_speed_limit * 0.9, scale=lane_speed_limit * 0.1),
    #         lane_speed_limit * 0.5, lane_speed_limit * 1.1
    #     )
    #
    #     # self.target_speed = np.clip(
    #     #     np.random.normal(loc=27.0, scale=2.5),
    #     #     20.0, 35.0
    #     # )
    #
    #     # --- 2. Randomize Aggressiveness (Time Headway T) ---
    #     self.TIME_WANTED = np.random.lognormal(mean=0.2, sigma=0.3)
    #     self.TIME_WANTED = np.clip(self.TIME_WANTED, 0.8, 2.5)
    #
    #     # --- 3. Randomize Jam Distance (s0) ---
    #     aggressiveness = 1.0 - (self.TIME_WANTED - 0.8) / (2.5 - 0.8)
    #     self.DISTANCE_WANTED = 1.0 + 2.0 * (1 - aggressiveness)
    #
    #     # --- 4. Randomize Acceleration Capability (a) ---
    #     self.COMFORT_ACC_MAX = np.random.triangular(left=1.0, mode=2.5, right=4.0)
    #
    #     # --- 5. Randomize Comfortable Deceleration (b) ---
    #     self.COMFORT_ACC_MIN = -np.random.uniform(1.5, 3.0)
    #
    #     # --- 6. Randomize Politeness (MOBIL p) ---
    #     self.POLITENESS = np.random.beta(a=2, b=5)
    #
    #     if aggressiveness > 0.6:
    #         self.POLITENESS *= 0.5
    #
    #     # --- 7. Randomize Lane Change Threshold (MOBIL Δa_th) ---
    #     self.LANE_CHANGE_MIN_ACC_GAIN = np.random.uniform(0.1, 0.3)
    #
    #     # --- 8. Randomize Delta parameter (IDM exponent) ---
    #     self.DELTA = np.random.choice([3, 4, 5], p=[0.2, 0.6, 0.2])

    def _randomize_parameters(self):
        """Randomize all driving parameters for intersection environment.

        Based on:
        - Kesting, Treiber & Helbing (2008) calibrated IDM parameters
        - Treiber & Kesting (2013) "Traffic Flow Dynamics", Ch. 11
        - Calibrated intersection IDM parameters (s0=2.0m, T~1.1s, a~2.0 m/s²)
          from signalized intersection studies

        Speed range aligned with intersection reward_speed_range: [7, 9] m/s
        """
        # --- 1. Desired Speed (v0) ---
        # Intersection speeds are 7-12 m/s (25-43 km/h), much lower than highway
        lane_speed_limit = self.lane.speed_limit
        self.target_speed = np.clip(
            np.random.normal(loc=lane_speed_limit * 0.85, scale=lane_speed_limit * 0.1),
            lane_speed_limit * 0.4, lane_speed_limit * 1.0
        )

        # --- 2. Time Headway T ---
        # Intersection T is ~29% lower than highway (1.1s vs 1.6s)
        # Range: 0.6s (aggressive) to 1.8s (cautious)
        self.TIME_WANTED = np.random.lognormal(mean=0.1, sigma=0.3)
        self.TIME_WANTED = np.clip(self.TIME_WANTED, 0.6, 1.8)

        # --- 3. Jam Distance (s0) ---
        # Literature: s0 = 2.0m for intersections
        # Range: 1.5m (aggressive) to 3.0m (cautious)
        aggressiveness = 1.0 - (self.TIME_WANTED - 0.6) / (1.8 - 0.6)
        self.DISTANCE_WANTED = 1.5 + 1.5 * (1 - aggressiveness)

        # --- 4. Max Acceleration (a) ---
        # ~28% higher at intersections than highways (2.0 vs 1.6 m/s²)
        # Vehicles need to accelerate from stops more decisively
        self.COMFORT_ACC_MAX = np.random.triangular(left=1.5, mode=2.5, right=4.0)

        # --- 5. Comfortable Deceleration (b) ---
        # Slightly higher for intersection stop-and-go
        self.COMFORT_ACC_MIN = -np.random.uniform(2.0, 4.0)

        # --- 6. Politeness (MOBIL p) ---
        # Less relevant at intersections (no lane changes), but kept for
        # consistency if MOBIL is ever queried
        self.POLITENESS = np.random.beta(a=2, b=5)
        if aggressiveness > 0.6:
            self.POLITENESS *= 0.5

        # --- 7. Lane Change Threshold ---
        # Not used in intersection (lateral=False) but set for safety
        self.LANE_CHANGE_MIN_ACC_GAIN = np.random.uniform(0.1, 0.3)

        # --- 8. IDM exponent (delta) ---
        self.DELTA = np.random.choice([3, 4, 5], p=[0.2, 0.6, 0.2])