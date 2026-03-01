"""
Testing and debugging script for IQ-Learn implementation

This script helps verify that everything is working correctly before
running full training.
"""

import torch
import numpy as np
import gymnasium as gym
import highway_env
from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper
from src.algorithms.iq_learner import IQLearnTrainer, ReplayBuffer, DQNNetwork


def test_environment_setup():
    """Test 1: Verify environment is set up correctly"""
    print("\n" + "="*60)
    print("TEST 1: Environment Setup")
    print("="*60)

    try:
        env = gym.make(ENV_CONFIG['id'], config=ENV_CONFIG)

        env = SVOPureWrapper(env, svo_alpha=0.0, lamb=1.0)

        state, _ = env.reset()
        print(f"✓ Environment created successfully")
        print(f"  State shape: {state.shape}")
        print(f"  State dim: {env.observation_space.shape[0]}")
        print(f"  Action space: {env.action_space} (n={env.action_space.n})")

        # Take a few random steps
        for i in range(5):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        env.close()
        print("✓ Environment test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}\n")
        return False


def test_replay_buffer():
    """Test 2: Verify replay buffer works"""
    print("="*60)
    print("TEST 2: Replay Buffer")
    print("="*60)

    try:
        buffer = ReplayBuffer(capacity=1000)

        # Add some fake transitions
        for i in range(100):
            state = np.random.randn(44)  # Example state
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            next_state = np.random.randn(44)
            done = np.random.rand() < 0.1
            buffer.add(state, action, reward, next_state, done)

        print(f"✓ Added 100 transitions to buffer")
        print(f"  Buffer size: {len(buffer)}")

        # Sample a batch
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch

        print(f"✓ Sampled batch of 32")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")

        print("✓ Replay buffer test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}\n")
        return False


def test_q_network():
    """Test 3: Verify Q-network forward pass"""
    print("="*60)
    print("TEST 3: Q-Network")
    print("="*60)

    try:
        env = gym.make(ENV_CONFIG['id'], config=ENV_CONFIG)

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n
        env.close()

        q_net = DQNNetwork(state_dim, action_dim, hidden_dims=[256, 256])

        # Forward pass
        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        q_values = q_net(states)

        print(f"✓ Q-network created")
        print(f"  Input shape: {states.shape}")
        print(f"  Output shape: {q_values.shape}")
        print(f"  Number of parameters: {sum(p.numel() for p in q_net.parameters())}")

        # Check shapes
        assert q_values.shape == (batch_size, action_dim), "Q-values shape mismatch"

        print("✓ Q-network test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Q-network test failed: {e}\n")
        return False


def test_iq_learn_trainer():
    """Test 4: Verify IQ-Learn trainer initialization"""
    print("="*60)
    print("TEST 4: IQ-Learn Trainer")
    print("="*60)

    try:
        # Create environment
        env = gym.make(ENV_CONFIG['id'], config=ENV_CONFIG)
        env = SVOPureWrapper(env, svo_alpha=0.0, lamb=1.0)

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n

        # Create trainer
        trainer = IQLearnTrainer(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],  # Smaller for testing
            lr=3e-4,
            device='cpu',  # Use CPU for testing
        )

        print(f"✓ Trainer created successfully")
        print(f"  Obs shape: {env.observation_space.shape} -> flattened state_dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Device: {trainer.device}")

        # Add some fake expert data — always use flat 1D arrays
        for i in range(50):
            state = np.random.randn(state_dim).reshape(env.observation_space.shape)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).reshape(env.observation_space.shape)
            done = False
            trainer.expert_buffer.add(state, action, reward, next_state, done)

        print(f"✓ Added {len(trainer.expert_buffer)} expert transitions")

        # Try one update
        info = trainer.update(batch_size=16)
        print(f"✓ Performed one update step")
        print(f"  Loss: {info['loss']:.4f}")
        print(f"  Expert Q mean: {info['expert_q_mean']:.4f}")

        # Test action selection
        state, _ = env.reset()
        action = trainer.select_action(state, epsilon=0.0)
        print(f"✓ Action selection works: action={action}")

        env.close()
        print("✓ IQ-Learn trainer test passed!\n")
        return True

    except Exception as e:
        print(f"✗ IQ-Learn trainer test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_demonstration_format():
    """Test 5: Verify demonstration format"""
    print("="*60)
    print("TEST 5: Demonstration Format")
    print("="*60)

    try:
        # Derive state_dim from the actual environment
        env = gym.make(ENV_CONFIG['id'], config=ENV_CONFIG)
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n
        env.close()

        trajectory = []

        for i in range(10):
            state = np.random.randn(state_dim)
            action = int(np.random.randint(0, action_dim))
            reward = float(np.random.randn())
            next_state = np.random.randn(state_dim)
            done = float(i == 9)  # Last step is done

            trajectory.append((state, action, reward, next_state, done))

        print(f"✓ Created trajectory with {len(trajectory)} transitions")

        # Verify format
        state, action, reward, next_state, done = trajectory[0]
        print(f"  State type: {type(state)}, shape: {state.shape}")
        print(f"  Action type: {type(action)}")
        print(f"  Reward type: {type(reward)}")
        print(f"  Next state type: {type(next_state)}, shape: {next_state.shape}")
        print(f"  Done type: {type(done)}")

        # Verify it can be added to buffer
        buffer = ReplayBuffer()
        for transition in trajectory:
            buffer.add(*transition)

        print(f"✓ Trajectory can be added to replay buffer")
        print(f"  Buffer size: {len(buffer)}")

        print("✓ Demonstration format test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Demonstration format test failed: {e}\n")
        return False


def test_save_load():
    """Test 6: Verify save/load functionality"""
    print("="*60)
    print("TEST 6: Save/Load")
    print("="*60)

    try:
        import tempfile

        # Create environment
        env = gym.make(ENV_CONFIG['id'], config=ENV_CONFIG)
        env = SVOPureWrapper(env, svo_alpha=0.0, lamb=1.0)

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n

        # Create trainer
        trainer1 = IQLearnTrainer(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            device='cpu',
        )

        # Get initial Q-values
        state = torch.randn(1, state_dim)
        q_values_before = trainer1.q_network(state).detach()

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        trainer1.save(temp_path)
        print(f"✓ Model saved to temporary file")

        # Create new trainer and load
        trainer2 = IQLearnTrainer(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            device='cpu',
        )

        trainer2.load(temp_path)
        print(f"✓ Model loaded from temporary file")

        # Verify Q-values match
        q_values_after = trainer2.q_network(state).detach()
        assert torch.allclose(q_values_before, q_values_after), "Q-values don't match after load"

        print(f"✓ Q-values match after save/load")

        # Cleanup
        import os
        os.remove(temp_path)
        env.close()

        print("✓ Save/load test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Save/load test failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60 + "\n")

    tests = [
        ("Environment Setup", test_environment_setup),
        ("Replay Buffer", test_replay_buffer),
        ("Q-Network", test_q_network),
        ("IQ-Learn Trainer", test_iq_learn_trainer),
        ("Demonstration Format", test_demonstration_format),
        ("Save/Load", test_save_load),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"Test '{name}' crashed: {e}\n")
            results.append((name, False))

    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")

    num_passed = sum(1 for _, passed in results if passed)
    num_total = len(results)

    print(f"\nTotal: {num_passed}/{num_total} tests passed")

    if num_passed == num_total:
        print("\n🎉 All tests passed! You're ready to train IQ-Learn.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")

    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()