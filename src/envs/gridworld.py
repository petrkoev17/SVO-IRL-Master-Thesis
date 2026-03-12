import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SVOIntersectionEnv(gym.Env):
    """
    A 5x5 deterministic gridworld designed to test Social Value Orientation.
    - Ego Agent (Learner): Starts at (0, 2), moves West to East.
    - Other Agent (Reactive): Starts at (2, 0), moves South to North.
    - Conflict Point: (2, 2)
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 5
        self.render_mode = render_mode

        # Action space: 0 = Wait, 1 = Forward
        self.action_space = spaces.Discrete(2)

        # Observation space: [ego_x, ego_y, other_x, other_y]
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start positions
        self.ego_pos = np.array([0, 2], dtype=np.float32)
        self.other_pos = np.array([2, 0], dtype=np.float32)

        self.step_count = 0
        self.max_steps = 20

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Track previous states
        ego_was_at_goal = (self.ego_pos[0] == self.grid_size - 1)
        other_was_at_goal = (self.other_pos[1] == self.grid_size - 1)

        # --- 1. Calculate Intended Moves ---
        next_ego_pos = self.ego_pos.copy()
        if action == 1 and not ego_was_at_goal:
            next_ego_pos[0] += 1

        next_other_pos = self.other_pos.copy()
        if not other_was_at_goal:
            next_other_pos[1] += 1

        # --- 2. Reactive Logic ---
        if (next_ego_pos[0] == next_other_pos[0]) and (next_ego_pos[1] == next_other_pos[1]):
            next_other_pos = self.other_pos.copy()

        self.ego_pos = next_ego_pos
        self.other_pos = next_other_pos

        # --- 3. Compute True Underlying Rewards ---
        ego_is_at_goal = (self.ego_pos[0] == self.grid_size - 1)
        other_is_at_goal = (self.other_pos[1] == self.grid_size - 1)

        # Goal rewards are strictly one-time events
        if ego_was_at_goal:
            r_self = 0.0
        elif ego_is_at_goal:
            r_self = 10.0
        else:
            r_self = -1.0

        if other_was_at_goal:
            r_global = 0.0
        elif other_is_at_goal:
            r_global = 10.0
        else:
            r_global = -1.0

        # --- 4. Termination ---
        terminated = bool(ego_is_at_goal)
        truncated = bool(self.step_count >= self.max_steps)

        if terminated and not other_is_at_goal:
            remaining_steps = (self.grid_size - 1) - self.other_pos[1]
            r_global -= remaining_steps

        reward = r_self

        info = {
            'rewards/component_self': float(r_self),
            'rewards/component_global': float(r_global),
            'collision': False,
            'ego_at_goal': ego_is_at_goal,
            'other_at_goal': other_is_at_goal
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate((self.ego_pos, self.other_pos))

    def render(self):
        if self.render_mode != "console":
            return

        grid = np.full((self.grid_size, self.grid_size), ".")
        grid[2, 2] = "X"  # Mark intersection

        ox, oy = int(self.other_pos[0]), int(self.other_pos[1])
        ex, ey = int(self.ego_pos[0]), int(self.ego_pos[1])

        if ox == ex and oy == ey:
            grid[oy, ox] = "C"  # Should theoretically never happen with reactive logic
        else:
            grid[oy, ox] = "O"
            grid[ey, ex] = "E"

        print(f"\nStep: {self.step_count}")
        for row in reversed(range(self.grid_size)):
            print(" ".join(grid[row]))
        print("-" * 10)