import numpy as np
import pickle
import os
from src.envs.gridworld import SVOIntersectionEnv


class SVOValueIteration:
    def __init__(self, env, gamma=0.99, svo_alpha=0.0):
        self.env = env
        self.gamma = gamma
        self.svo_alpha = svo_alpha
        self.grid_size = env.grid_size

        # SVO components
        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)

        # Initialize Value table for all states (ex, ey, ox, oy)
        self.V = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size))

        # Actions: 0 = Wait, 1 = Forward
        self.actions = [0, 1]

    def _simulate_step(self, ex, ey, ox, oy, action):
        """Simulates the environment logic and prevents reward farming."""
        next_ex, next_ey = ex, ey
        next_ox, next_oy = ox, oy

        # Track if they were ALREADY at the goal before moving
        ego_was_at_goal = (ex == self.grid_size - 1)
        other_was_at_goal = (oy == self.grid_size - 1)

        # 1. Intended moves
        if action == 1 and not ego_was_at_goal:
            next_ex += 1
        if not other_was_at_goal:
            next_oy += 1

        # 2. Reactive logic (Other yields if conflict)
        if next_ex == next_ox and next_ey == next_oy:
            next_ox, next_oy = ox, oy

            # 3. Compute rewards (Only reward exactly upon arrival)
        ego_is_at_goal = (next_ex == self.grid_size - 1)
        other_is_at_goal = (next_oy == self.grid_size - 1)

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

        # 4. Termination
        done = bool(ego_is_at_goal)

        if done and not other_is_at_goal:
            remaining_steps = (self.grid_size - 1) - next_oy
            r_global -= remaining_steps

        return next_ex, next_ey, next_ox, next_oy, r_self, r_global, done

    def _compute_svo_reward(self, r_self, r_global):
        return self.cos_alpha * r_self + self.sin_alpha * r_global

    def solve(self, threshold=1e-5, max_iterations=1000):
        """Runs Value Iteration until convergence."""
        print(f"Running Value Iteration for SVO alpha = {self.svo_alpha:.2f} rad...")

        for iteration in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)

            # Sweep through the entire state space
            for ex in range(self.grid_size):
                for ey in range(self.grid_size):
                    for ox in range(self.grid_size):
                        for oy in range(self.grid_size):

                            # Skip if ego is already at the goal (terminal state)
                            if ex == self.grid_size - 1:
                                new_V[ex, ey, ox, oy] = 0
                                continue

                            q_values = []
                            for a in self.actions:
                                nx, ny, nox, noy, r_self, r_global, done = self._simulate_step(ex, ey, ox, oy, a)
                                r_svo = self._compute_svo_reward(r_self, r_global)

                                if done:
                                    q_val = r_svo
                                else:
                                    q_val = r_svo + self.gamma * self.V[nx, ny, nox, noy]
                                q_values.append(q_val)

                            best_value = max(q_values)
                            delta = max(delta, abs(best_value - self.V[ex, ey, ox, oy]))
                            new_V[ex, ey, ox, oy] = best_value

            self.V = new_V
            if delta < threshold:
                print(f"Converged in {iteration + 1} iterations.")
                break

    def get_optimal_action(self, ex, ey, ox, oy):
        """Returns the best action based on the converged V-table."""
        best_action = None
        best_q = -float('inf')

        for a in self.actions:
            nx, ny, nox, noy, r_self, r_global, done = self._simulate_step(ex, ey, ox, oy, a)
            r_svo = self._compute_svo_reward(r_self, r_global)

            if done:
                q_val = r_svo
            else:
                q_val = r_svo + self.gamma * self.V[nx, ny, nox, noy]

            if q_val > best_q:
                best_q = q_val
                best_action = a

        return best_action

    def extract_trajectories(self, num_trajectories=10):
        """Rolls out the optimal policy to generate the dataset."""
        trajectories = []

        for _ in range(num_trajectories):
            obs, _ = self.env.reset()
            trajectory = []
            done = False

            while not done:
                ex, ey, ox, oy = map(int, obs)
                action = self.get_optimal_action(ex, ey, ox, oy)

                next_obs, standard_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                r_self = info['rewards/component_self']
                r_global = info['rewards/component_global']
                crashed = info['collision']

                # Format: (s, a, r, s', done, crashed, r_self, r_global)
                transition = (
                    obs.copy(),
                    action,
                    float(r_self),
                    next_obs.copy(),
                    done,
                    crashed,
                    float(r_self),
                    float(r_global)
                )
                trajectory.append(transition)
                obs = next_obs

            trajectories.append(trajectory)

        return trajectories


def main():
    os.makedirs('expert_data', exist_ok=True)
    env = SVOIntersectionEnv()

    # Define SVO profiles to extract
    svo_profiles = {
        'egoistic': 0.0,
        'prosocial': np.pi / 4,
        'altruistic': np.pi / 2
    }

    for name, alpha in svo_profiles.items():
        solver = SVOValueIteration(env, gamma=0.99, svo_alpha=alpha)
        solver.solve()

        # We generate 10 identical trajectories to form a small batch
        trajectories = solver.extract_trajectories(num_trajectories=10)

        file_path = f'./expert_demonstrations/expert_{name}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(trajectories, f)

        print(f"Saved {len(trajectories)} {name} trajectories to {file_path}")

        # Print one trajectory out just to verify the behaviour
        print(f"--- Example {name.capitalize()} Trajectory ---")
        for step, t in enumerate(trajectories[0]):
            action_name = "Forward" if t[1] == 1 else "Wait"
            print(f"Step {step}: Action={action_name}, r_self={t[6]:.1f}, r_global={t[7]:.1f}")
        print("\n")


if __name__ == "__main__":
    main()