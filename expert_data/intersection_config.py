"""
Configuration file for the Intersection environment.

Reward structure (from IntersectionEnv._agent_rewards):
    - collision_reward: penalty for crashing (default: -5)
    - high_speed_reward: reward for maintaining speed in [7, 9] m/s range
    - arrived_reward: reward for reaching destination (overrides other rewards)
    - on_road_reward: multiplicative, zeroes reward if off-road

Note: There is NO right_lane_reward in the intersection environment.
The SVOPureWrapper's _calculate_neighbourhood_reward must be adapted
to handle this (skip right_lane_reward, add arrived_reward).
"""

INTERSECTION_CONFIG = {
    "id": "intersection-v1",

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
        "absolute": True,
        "flatten": True,
        "observe_intentions": False,
    },

    # Actions: only longitudinal (speed control)
    # Lateral steering is automatic via lane-following controller
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 4.5, 9],
    },

    # Traffic
    # "other_vehicles_type": "src.vehicles.randomized_idm_vehicle.RandomizedIDMVehicle",
    "destination": "o1",
    "controlled_vehicles": 1,
    "initial_vehicle_count": 5,
    "spawn_probability": 0.3,
    "duration": 20,

    # Physics
    "simulation_frequency": 10,
    "policy_frequency": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "show_trajectories": False,
    "render_agent": False,
    "offscreen_rendering": False,

    # Reward weights
    "collision_reward": -5.0,
    "high_speed_reward": 1.0,
    "arrived_reward": 1.0,
    "reward_speed_range": [7.0, 9.0],
    "normalize_reward": False,

    "offroad_terminal": False,
}