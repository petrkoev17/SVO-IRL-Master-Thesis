"""
Configuration file for the Intersection environment.

Reward structure v2 — progress-based, SVO-compatible.

Motivation:
    The original reward (collision_reward=-5, high_speed_reward=1, arrived_reward=1)
    creates a 5:1 penalty-to-reward ratio that makes stopping optimal for any
    prosocial SVO angle. Literature (Schwarting 2019, Tong 2024, hierarchical
    reward designs) shows that intersection rewards need a strong continuous
    progress signal so that cooperative agents learn "drive carefully" rather
    than "don't drive."

Changes from v1:
    - NEW: progress_reward (w=2.0) — continuous reward for longitudinal
      advance along the lane toward the exit. Dominant positive signal.
    - collision_reward reduced from -5 to -2 (ratio ~2:1 with progress).
    - high_speed_reward reduced to 0.5 (supplementary, not dominant).
    - arrived_reward kept at 1.0 as a terminal bonus.
    - NEW: timeout_reward = -0.5 — mild penalty for running out of time
      without arriving or crashing, to discourage infinite waiting.

The progress reward is computed in the SVOIntersectionWrapper, not by
highway-env's built-in reward, so the env's own reward structure is
replaced entirely by the wrapper.

Note: There is NO right_lane_reward in the intersection environment.
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

    # Traffic (reduced density for solvable intersection)
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
    "other_vehicles_type": "src.vehicles.randomized_idm_vehicle.RandomizedIDMVehicle",

    # ============================================================
    # Reward weights (v2 — progress-based, SVO-compatible)
    # ============================================================
    # These are read by SVOIntersectionWrapper, NOT by highway-env's
    # built-in reward. The wrapper computes its own reward.
    "collision_reward": -2.0,
    "progress_reward": 2.0,
    "high_speed_reward": 0.5,
    "arrived_reward": 1.0,
    "timeout_reward": -0.5,
    "reward_speed_range": [7.0, 9.0],
    "normalize_reward": False,

    # Progress reward normalisation distance (meters).
    # Lane longitudinal coordinates go from 0 to ~exit_distance.
    # progress = delta_longitudinal / reference_distance per step.
    "exit_distance": 25.0,

    # Neighbour search radius for r_global computation (meters).
    "neighbour_distance": 30.0,

    "offroad_terminal": False,
}