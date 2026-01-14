import numpy as np

"""
Configuration file for the environment settings.
"""

ENV_CONFIG = {
    "id": "highway-v0",

    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": False,
        "order": "sorted",
    },

    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "acceleration_range": [-5.0, 5.0],
        "steering_range": [-np.pi/4, np.pi/4],
    },

    # Traffic
    "lanes_count": 4,
    "vehicles_count": 25,
    "duration": 40,
    "initial_spacing": 2.0,
    "controlled_vehicles": 1,

    # Physics
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": False,
    "offscreen_rendering": False,

    # SVO Reward Weights
    "collision_reward": -1.0,
    "high_speed_reward": 1.0,
    "reward_speed_range": [20.0, 30.0],
    "right_lane_reward": 0.5,
    "on_road_reward": 1.5,
    "off_road_penalty": 0,

}
