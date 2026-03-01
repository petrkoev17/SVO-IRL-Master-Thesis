"""
Absorbing State Wrapper for Highway Environment

When the agent crashes (done=True), transitions to a special absorbing state
that loops back to itself. This helps the discriminator distinguish:
- Normal trajectories: state → action → next_state
- Crashed trajectories: state → action → ABSORBING → ABSORBING → ...

This is SVO-safe because:
- Doesn't bias toward cautious vs aggressive driving
- Just makes terminal states identifiable
- Different SVO values can still prefer different risk/reward tradeoffs
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AbsorbingWrapper(gym.Wrapper):
    """
    Wraps an environment to add absorbing states.

    After a terminal state (done=True), the environment transitions to an
    absorbing state that maps to itself with zero reward.

    This is a standard technique in IRL to help discriminators learn that
    early termination (crashes) are different from successful trajectories.
    """

    def __init__(self, env):
        super().__init__(env)

        # Add one dimension to observation space for absorbing indicator
        if isinstance(env.observation_space, spaces.Box):
            low = np.append(env.observation_space.low, 0.0)
            high = np.append(env.observation_space.high, 1.0)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=env.observation_space.dtype
            )
        else:
            raise ValueError("AbsorbingWrapper only supports Box observation spaces")

        self.absorbing_state = None
        self._in_absorbing_state = False

    def _add_absorbing_indicator(self, obs, is_absorbing):
        """Add absorbing indicator to observation."""
        indicator = np.array([1.0 if is_absorbing else 0.0], dtype=obs.dtype)
        return np.append(obs, indicator)

    def _get_absorbing_state(self):
        """
        Create the absorbing state.
        Uses zeros with absorbing indicator = 1.
        """
        if self.absorbing_state is None:
            # Create a zero observation
            zero_obs = np.zeros(self.env.observation_space.shape, dtype=self.env.observation_space.dtype)
            # Mark as absorbing
            self.absorbing_state = self._add_absorbing_indicator(zero_obs, is_absorbing=True)
        return self.absorbing_state.copy()

    def reset(self, **kwargs):
        """Reset environment and clear absorbing state flag."""
        obs, info = self.env.reset(**kwargs)
        self._in_absorbing_state = False
        # Add non-absorbing indicator
        obs_with_indicator = self._add_absorbing_indicator(obs, is_absorbing=False)
        return obs_with_indicator, info

    def step(self, action):
        """
        Step environment. If in absorbing state, stay there.
        If transitioning to terminal, move to absorbing state.
        """
        if self._in_absorbing_state:
            # Already in absorbing state - stay there with zero reward
            obs = self._get_absorbing_state()
            reward = 0.0
            terminated = True
            truncated = False
            info = {"absorbing_state": True}
            return obs, reward, terminated, truncated, info

        # Normal step
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done:
            # Transition to absorbing state
            self._in_absorbing_state = True
            obs = self._get_absorbing_state()
            info["absorbing_state"] = True
        else:
            # Add non-absorbing indicator to normal observation
            obs = self._add_absorbing_indicator(obs, is_absorbing=False)
            info["absorbing_state"] = False

        return obs, reward, terminated, truncated, info



def test_absorbing_wrapper():
    """Test the absorbing wrapper to make sure it works correctly."""
    import gymnasium as gym

    print("Testing AbsorbingWrapper...")

    # Create a simple environment
    env = gym.make("CartPole-v1")
    wrapped_env = AbsorbingWrapper(env)

    print(f"Original obs space: {env.observation_space}")
    print(f"Wrapped obs space: {wrapped_env.observation_space}")
    print(f"Original obs shape: {env.observation_space.shape}")
    print(f"Wrapped obs shape: {wrapped_env.observation_space.shape}")

    # Test reset
    obs, info = wrapped_env.reset(seed=42)
    print(f"\nAfter reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Absorbing indicator: {obs[-1]}")
    print(f"  Should be 0.0: {obs[-1] == 0.0}")

    # Test normal step
    obs, reward, terminated, truncated, info = wrapped_env.step(0)
    print(f"\nAfter normal step:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Absorbing indicator: {obs[-1]}")
    print(f"  Done: {terminated or truncated}")

    # Force termination by running until done
    for _ in range(500):
        obs, reward, terminated, truncated, info = wrapped_env.step(env.action_space.sample())
        if terminated or truncated:
            break

    print(f"\nAfter termination:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Absorbing indicator: {obs[-1]}")
    print(f"  Should be 1.0: {obs[-1] == 1.0}")
    print(f"  In absorbing state: {info.get('absorbing_state', False)}")

    # Test absorbing state persistence
    obs2, reward2, terminated2, truncated2, info2 = wrapped_env.step(0)
    print(f"\nAfter step in absorbing state:")
    print(f"  Observation shape: {obs2.shape}")
    print(f"  Absorbing indicator: {obs2[-1]}")
    print(f"  Should be 1.0: {obs2[-1] == 1.0}")
    print(f"  Reward should be 0.0: {reward2}")
    print(f"  Still done: {terminated2 or truncated2}")
    print(f"  Observations match: {np.allclose(obs, obs2)}")

    env.close()
    wrapped_env.close()

    print("\n✓ AbsorbingWrapper test complete!")


if __name__ == "__main__":
    test_absorbing_wrapper()