"""
Extract expert demonstrations from trained DQN agents
Collects trajectories for IQ-Learn training
"""

import gymnasium as gym
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from stable_baselines3 import DQN
from tqdm import tqdm
import os


def extract_demonstrations_from_agent(
    agent: DQN,
    env: gym.Env,
    num_episodes: int = 100,
    deterministic: bool = True,
    max_steps_per_episode: int = 1000,
    verbose: bool = True
) -> Tuple[List[List[Tuple]], Dict[str, float]]:
    """
    Extract demonstrations from a trained DQN agent

    Args:
        agent: Trained DQN agent
        env: Environment to collect trajectories in
        num_episodes: Number of episodes to collect
        deterministic: Whether to use deterministic actions
        max_steps_per_episode: Maximum steps per episode
        verbose: Whether to print progress

    Returns:
        trajectories: List of trajectories, each trajectory is a list of (s, a, r, s', done)
        stats: Dictionary of statistics about the collected demonstrations
    """
    trajectories = []
    all_rewards = []
    all_lengths = []
    collision_count = 0

    iterator = tqdm(range(num_episodes)) if verbose else range(num_episodes)

    for episode in iterator:
        trajectory = []
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        for step in range(max_steps_per_episode):
            # Get action from agent
            action, _ = agent.predict(state, deterministic=deterministic)

            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            trajectory.append((
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                float(done),
                bool(info.get('crashed', False)),
            ))

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        # Check for collision
        if terminated and hasattr(env.unwrapped, 'vehicle'):
            if env.unwrapped.vehicle.crashed:
                collision_count += 1

        trajectories.append(trajectory)
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        if verbose and episode % 10 == 0:
            iterator.set_description(
                f"Ep {episode}: Reward={episode_reward:.2f}, Length={episode_length}, "
                f"Avg Reward={np.mean(all_rewards[-10:]):.2f}"
            )

    # Compute statistics
    stats = {
        'num_episodes': num_episodes,
        'total_transitions': sum(len(traj) for traj in trajectories),
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_length': float(np.mean(all_lengths)),
        'std_length': float(np.std(all_lengths)),
        'collision_rate': float(collision_count / num_episodes),
        'min_reward': float(np.min(all_rewards)),
        'max_reward': float(np.max(all_rewards)),
    }

    return trajectories, stats


def save_demonstrations(
    trajectories: List[List[Tuple]],
    stats: Dict[str, float],
    save_path: str,
    metadata: Optional[Dict] = None
):
    """Save demonstrations to disk"""
    data = {
        'trajectories': trajectories,
        'stats': stats,
        'metadata': metadata or {}
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved {len(trajectories)} trajectories to {save_path}")
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"Mean reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Collision rate: {stats['collision_rate']:.2%}")


def load_demonstrations(load_path: str) -> Tuple[List[List[Tuple]], Dict[str, float], Dict]:
    """Load demonstrations from disk"""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    return data['trajectories'], data['stats'], data.get('metadata', {})


def extract_from_multiple_agents(
    agent_paths: Dict[str, str],
    env_fn,  # Single callable OR dict mapping agent_name -> callable
    num_episodes_per_agent: int = 100,
    save_dir: str = './expert_demonstrations',
    deterministic: bool = True
) -> Dict[str, str]:
    """
    Extract demonstrations from multiple agents with different SVO angles

    Args:
        agent_paths: Dictionary mapping agent names to model paths
        env_fn: Either a single callable (same env for all agents) OR a dict
                mapping agent_name -> callable, so each agent gets its own
                environment with the correct SVO angle in radians.
        num_episodes_per_agent: Number of episodes to collect per agent
        save_dir: Directory to save demonstrations
        deterministic: Whether to use deterministic actions

    Returns:
        demo_paths: Dictionary mapping agent names to saved demonstration paths
    """
    os.makedirs(save_dir, exist_ok=True)
    demo_paths = {}

    for agent_name, agent_path in agent_paths.items():
        print(f"\n{'='*60}")
        print(f"Extracting demonstrations from: {agent_name}")
        print(f"Model path: {agent_path}")
        print(f"{'='*60}")

        # Resolve env factory: per-agent dict or shared callable
        if isinstance(env_fn, dict):
            agent_env_fn = env_fn[agent_name]
        else:
            agent_env_fn = env_fn

        # Create environment
        env = agent_env_fn()

        # Load agent
        agent = DQN.load(agent_path)

        # Extract demonstrations
        trajectories, stats = extract_demonstrations_from_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes_per_agent,
            deterministic=deterministic,
            verbose=True
        )

        # Save demonstrations
        save_path = os.path.join(save_dir, f"{agent_name}_demonstrations.pkl")
        metadata = {
            'agent_name': agent_name,
            'agent_path': agent_path,
            'deterministic': deterministic
        }
        save_demonstrations(trajectories, stats, save_path, metadata)

        demo_paths[agent_name] = save_path
        env.close()

    return demo_paths


# def combine_demonstrations(
#     demo_paths: List[str],
#     output_path: str,
#     weights: Optional[List[float]] = None
# ) -> Tuple[List[List[Tuple]], Dict[str, float]]:
#     """
#     Combine demonstrations from multiple sources
#
#     Args:
#         demo_paths: List of paths to demonstration files
#         output_path: Path to save combined demonstrations
#         weights: Optional weights for sampling from each source (must sum to 1)
#
#     Returns:
#         combined_trajectories: Combined list of trajectories
#         combined_stats: Statistics of combined dataset
#     """
#     if weights is not None:
#         assert len(weights) == len(demo_paths), "Weights must match number of demo sources"
#         assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
#     else:
#         weights = [1.0 / len(demo_paths)] * len(demo_paths)
#
#     all_trajectories = []
#     all_stats = []
#
#     # Load all demonstrations
#     for path in demo_paths:
#         trajectories, stats, metadata = load_demonstrations(path)
#         all_trajectories.append(trajectories)
#         all_stats.append(stats)
#         print(f"Loaded {len(trajectories)} trajectories from {path}")
#
#     # Combine based on weights
#     combined_trajectories = []
#     for trajs, weight in zip(all_trajectories, weights):
#         num_samples = int(len(trajs) * weight * len(demo_paths))
#         sampled_indices = np.random.choice(len(trajs), size=num_samples, replace=True)
#         sampled_trajs = [trajs[i] for i in sampled_indices]
#         combined_trajectories.extend(sampled_trajs)
#
#     # Compute combined statistics
#     all_rewards = []
#     all_lengths = []
#     collision_count = 0
#
#     for traj in combined_trajectories:
#         traj_reward = sum(t[2] for t in traj)
#         all_rewards.append(traj_reward)
#         all_lengths.append(len(traj))
#
#         # Check if last transition is a collision
#         if any(t[5] for t in traj):
#             collision_count += 1
#
#     combined_stats = {
#         'num_episodes': len(combined_trajectories),
#         'total_transitions': sum(len(traj) for traj in combined_trajectories),
#         'mean_reward': float(np.mean(all_rewards)),
#         'std_reward': float(np.std(all_rewards)),
#         'mean_length': float(np.mean(all_lengths)),
#         'std_length': float(np.std(all_lengths)),
#         'collision_rate': float(collision_count / len(combined_trajectories)),
#     }
#
#     # Save combined demonstrations
#     metadata = {
#         'source_paths': demo_paths,
#         'weights': weights,
#         'num_sources': len(demo_paths)
#     }
#     save_demonstrations(combined_trajectories, combined_stats, output_path, metadata)
#
#     return combined_trajectories, combined_stats


def combine_demonstrations(
        demo_paths: List[str],
        output_path: str,
        weights: Optional[List[float]] = None
) -> Tuple[List[List[Tuple]], Dict[str, float]]:
    """
    Combine demonstrations using saved stats for means, but re-computing
    STD for both Reward and Length from the sampled trajectories.
    """
    if weights is not None:
        assert len(weights) == len(demo_paths), "Weights must match number of demo sources"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    else:
        weights = [1.0 / len(demo_paths)] * len(demo_paths)

    combined_trajectories = []
    weighted_mean_reward = 0.0
    weighted_mean_length = 0.0
    weighted_collision_rate = 0.0
    total_transitions = 0

    for path, weight in zip(demo_paths, weights):
        trajectories, stats, metadata = load_demonstrations(path)

        num_to_sample = int(len(trajectories) * weight * len(demo_paths))
        sampled_indices = np.random.choice(len(trajectories), size=num_to_sample, replace=True)
        sampled_trajs = [trajectories[i] for i in sampled_indices]

        combined_trajectories.extend(sampled_trajs)

        weighted_mean_reward += stats['mean_reward'] * weight
        weighted_mean_length += stats['mean_length'] * weight
        weighted_collision_rate += stats['collision_rate'] * weight
        total_transitions += sum(len(t) for t in sampled_trajs)

    combined_episode_rewards = [sum(t[2] for t in traj) for traj in combined_trajectories]
    combined_episode_lengths = [len(traj) for traj in combined_trajectories]

    true_std_reward = float(np.std(combined_episode_rewards))
    true_std_length = float(np.std(combined_episode_lengths))

    combined_stats = {
        'num_episodes': len(combined_trajectories),
        'total_transitions': total_transitions,
        'mean_reward': weighted_mean_reward,
        'std_reward': true_std_reward,
        'mean_length': weighted_mean_length,
        'std_length': true_std_length,  # <--- Added true STD of length
        'collision_rate': weighted_collision_rate,
    }

    # Save
    metadata = {
        'source_paths': demo_paths,
        'weights': weights,
        'calculation_method': "weighted_stats_with_sampled_stds"
    }
    save_demonstrations(combined_trajectories, combined_stats, output_path, metadata)

    return combined_trajectories, combined_stats


if __name__ == "__main__":
    # Example usage
    print("This is a utility module for extracting demonstrations.")
    print("Import and use the functions in your training script.")