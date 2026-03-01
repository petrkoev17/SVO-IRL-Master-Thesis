"""
Fixed Horizon Wrapper for AIRL Training

Ensures all episodes run for exactly a fixed number of steps by:
1. Continuing episodes after termination in an absorbing state
2. Only signaling done=True at the fixed horizon
3. Preserving crash information via absorbing state indicator

This allows using allow_variable_horizon=False while maintaining
information about when the agent actually crashed.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FixedHorizonWrapper(gym.Wrapper):
    """
    Wrapper that enforces a fixed horizon for all episodes.

    After the environment terminates (crash or goal), the wrapper:
    - Transitions to an absorbing state
    - Continues for remaining steps until fixed horizon
    - Returns the absorbing state for all subsequent observations
    - Only signals done=True at the fixed horizon step

    This is essential for allow_variable_horizon=False in AIRL.
    """

    def __init__(self, env, horizon):
        """
        Args:
            env: The environment to wrap
            horizon: Fixed number of steps per episode
        """
        super().__init__(env)
        self.horizon = horizon
        self.step_count = 0
        self.episode_terminated = False
        self.absorbing_state = None

        # Modify observation space to include absorbing indicator
        if isinstance(env.observation_space, spaces.Box):
            low = np.append(env.observation_space.low, 0.0)
            high = np.append(env.observation_space.high, 1.0)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=env.observation_space.dtype
            )
        else:
            raise ValueError("FixedHorizonWrapper only supports Box observation spaces")

    def _add_absorbing_indicator(self, obs, is_absorbing):
        """Add absorbing indicator to observation."""
        indicator = np.array([1.0 if is_absorbing else 0.0], dtype=obs.dtype)
        return np.append(obs, indicator)

    def _get_absorbing_state(self):
        """Create the absorbing state (zeros with indicator=1)."""
        if self.absorbing_state is None:
            zero_obs = np.zeros(self.env.observation_space.shape, dtype=self.env.observation_space.dtype)
            self.absorbing_state = self._add_absorbing_indicator(zero_obs, is_absorbing=True)
        return self.absorbing_state.copy()

    def reset(self, **kwargs):
        """Reset environment and step counter."""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.episode_terminated = False

        # Add non-absorbing indicator
        obs_with_indicator = self._add_absorbing_indicator(obs, is_absorbing=False)
        return obs_with_indicator, info

    def step(self, action):
        """
        Step environment, enforcing fixed horizon.

        If episode has terminated, stay in absorbing state.
        Only signal done=True when reaching the fixed horizon.
        """
        self.step_count += 1

        if self.episode_terminated:
            # Already terminated, stay in absorbing state
            obs = self._get_absorbing_state()
            reward = 0.0
            terminated = False  # Don't signal done yet
            truncated = False
            info = {"absorbing_state": True, "fixed_horizon_padding": True}
        else:
            # Normal step
            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                # Episode terminated, transition to absorbing state
                self.episode_terminated = True
                obs = self._get_absorbing_state()
                info["absorbing_state"] = True
                info["actual_termination_step"] = self.step_count
                # Don't signal done yet - wait for horizon
                terminated = False
                truncated = False
            else:
                # Add non-absorbing indicator
                obs = self._add_absorbing_indicator(obs, is_absorbing=False)
                info["absorbing_state"] = False

        # Only signal done when reaching the fixed horizon
        if self.step_count >= self.horizon:
            terminated = True
            info["fixed_horizon_reached"] = True

        return obs, reward, terminated, truncated, info


def test_fixed_horizon_wrapper():
    """Test the fixed horizon wrapper."""
    import gymnasium as gym

    print("Testing FixedHorizonWrapper...")

    # Create a simple environment
    env = gym.make("CartPole-v1")
    wrapped_env = FixedHorizonWrapper(env, horizon=20)

    print(f"Original obs space: {env.observation_space}")
    print(f"Wrapped obs space: {wrapped_env.observation_space}")
    print(f"Fixed horizon: {wrapped_env.horizon}")

    # Test episode that crashes early
    print("\n=== Test 1: Episode that terminates early ===")
    obs, info = wrapped_env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}, absorbing indicator: {obs[-1]}")

    step_count = 0
    total_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        step_count += 1
        total_reward += reward

        is_absorbing = info.get("absorbing_state", False)
        print(f"Step {step_count}: reward={reward:.2f}, done={terminated or truncated}, "
              f"absorbing={is_absorbing}, obs[-1]={obs[-1]:.1f}")

        if terminated or truncated:
            print(f"Episode finished at step {step_count}")
            print(f"Total reward: {total_reward:.2f}")
            break

    assert step_count == wrapped_env.horizon, f"Episode should run for exactly {wrapped_env.horizon} steps"
    print(f"✓ Episode ran for exactly {wrapped_env.horizon} steps as expected")

    # Test that multiple episodes all have same length
    print("\n=== Test 2: Multiple episodes all same length ===")
    lengths = []
    for ep in range(5):
        obs, info = wrapped_env.reset(seed=ep)
        step_count = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            step_count += 1
            if terminated or truncated:
                break
        lengths.append(step_count)
        print(f"Episode {ep + 1}: length={step_count}")

    assert all(l == wrapped_env.horizon for l in lengths), "All episodes should have same length"
    print(f"✓ All episodes have length {wrapped_env.horizon}")

    env.close()
    wrapped_env.close()

    print("\n✓ FixedHorizonWrapper test complete!")


if __name__ == "__main__":
    test_fixed_horizon_wrapper()