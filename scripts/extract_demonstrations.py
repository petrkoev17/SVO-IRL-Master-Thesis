"""
Extract expert demonstrations from trained DQN agents
Collects trajectories for IQ-Learn training

Each transition is stored as a 8-element tuple:
    (state, action, reward, next_state, done, crashed, r_self, r_global)

r_self and r_global are the raw SVO reward components from the wrapper,
independent of whatever α was used during collection. They enable
SVO-regularized IQ-Learn to recompute R_SVO with any target angle.
"""

import gymnasium as gym
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from stable_baselines3 import DQN
from tqdm import tqdm
import os

def compute_cumulative_svo_returns(
    trajectories: List[List[Tuple]],
    gamma_svo: float = 0.99,
) -> List[List[Tuple]]:
    """
    Compute cumulative discounted returns for r_self and r_global,
    then augment each transition with G_self and G_global.

    G_self_t  = sum_{k=0}^{T-1-t}  gamma_svo^k * r_self_{t+k}
    G_global_t = sum_{k=0}^{T-1-t}  gamma_svo^k * r_global_{t+k}

    Input:  8-element tuples (s, a, r, s', done, crashed, r_self, r_global)
    Output: 10-element tuples (..., G_self, G_global)
    """
    augmented = []

    for traj in trajectories:
        T = len(traj)
        if T == 0:
            augmented.append([])
            continue

        if len(traj[0]) < 8:
            raise ValueError(
                f"Transitions must have >= 8 elements (got {len(traj[0])}). "
                f"Re-extract demos with SVO wrapper."
            )

        r_selfs = np.array([float(t[6]) for t in traj], dtype=np.float64)
        r_globals = np.array([float(t[7]) for t in traj], dtype=np.float64)

        G_selfs = np.zeros(T, dtype=np.float64)
        G_globals = np.zeros(T, dtype=np.float64)

        G_selfs[-1] = r_selfs[-1]
        G_globals[-1] = r_globals[-1]

        for t in range(T - 2, -1, -1):
            G_selfs[t] = r_selfs[t] + gamma_svo * G_selfs[t + 1]
            G_globals[t] = r_globals[t] + gamma_svo * G_globals[t + 1]

        aug_traj = []
        for t_idx, transition in enumerate(traj):
            base = transition[:8]
            aug_traj.append(base + (
                float(G_selfs[t_idx]),
                float(G_globals[t_idx]),
            ))
        augmented.append(aug_traj)

    return augmented


def extract_demonstrations_from_agent(
    agent: DQN,
    env: gym.Env,
    num_episodes: int = 100,
    deterministic: bool = True,
    max_steps_per_episode: int = 1000,
    verbose: bool = True
) -> Tuple[List[List[Tuple]], Dict[str, float]]:
    """
    Extract demonstrations from a trained DQN agent.

    Returns:
        trajectories: List of trajectories, each trajectory is a list of
                      (s, a, r, s', done, crashed, r_self, r_global)
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

            # Extract SVO reward components (available from SVOPureWrapper)
            r_self = float(info.get('rewards/component_self', 0.0))
            r_global = float(info.get('rewards/component_global', 0.0))

            # Store transition with SVO components
            trajectory.append((
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                float(done),
                bool(info.get('crashed', False)),
                r_self,
                r_global,
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
    env_fn,
    num_episodes_per_agent: int = 100,
    save_dir: str = './expert_demonstrations',
    deterministic: bool = True
) -> Dict[str, str]:
    """
    Extract demonstrations from multiple agents with different SVO angles.

    Args:
        agent_paths: Dictionary mapping agent names to model paths
        env_fn: Either a single callable (same env for all agents) OR a dict
                mapping agent_name -> callable
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

        # Resolve env factory
        if isinstance(env_fn, dict):
            agent_env_fn = env_fn[agent_name]
        else:
            agent_env_fn = env_fn

        env = agent_env_fn()
        agent = DQN.load(agent_path)

        trajectories, stats = extract_demonstrations_from_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes_per_agent,
            deterministic=deterministic,
            verbose=True
        )

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


def compute_stats_from_trajectories(
    trajectories: List[List[Tuple]],
) -> Dict[str, float]:
    """
    Compute statistics directly from a list of trajectories.

    Works with any tuple length >= 5. Collision detection uses:
        - index [5] (crashed) for 6- and 8-element tuples
        - not available for 5-element tuples
    """
    all_rewards = []
    all_lengths = []
    collision_count = 0

    for traj in trajectories:
        traj_reward = sum(t[2] for t in traj)
        all_rewards.append(traj_reward)
        all_lengths.append(len(traj))

        # Check for collision — 'crashed' is at index [5] in both
        # the legacy 6-element and current 8-element formats
        if len(traj[0]) >= 6:
            if any(t[5] for t in traj):
                collision_count += 1

    n = len(trajectories)
    return {
        'num_episodes': n,
        'total_transitions': sum(all_lengths),
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_length': float(np.mean(all_lengths)),
        'std_length': float(np.std(all_lengths)),
        'collision_rate': float(collision_count / n) if n > 0 else 0.0,
    }


def combine_demonstrations(
        demo_paths: List[str],
        output_path: str,
        weights: Optional[List[float]] = None
) -> Tuple[List[List[Tuple]], Dict[str, float]]:
    """
    Combine demonstrations from multiple sources by sampling according
    to the given weights. All statistics are computed from the actual
    sampled trajectories (not from the original per-source stats).

    Args:
        demo_paths: List of paths to demonstration .pkl files.
        output_path: Path to save the combined .pkl file.
        weights: Sampling weights per source (must sum to 1).
                 Defaults to uniform.

    Returns:
        combined_trajectories, combined_stats
    """
    if weights is not None:
        assert len(weights) == len(demo_paths), "Weights must match number of demo sources"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    else:
        weights = [1.0 / len(demo_paths)] * len(demo_paths)

    combined_trajectories = []

    for path, weight in zip(demo_paths, weights):
        trajectories, stats, _ = load_demonstrations(path)
        print(f"Loaded {len(trajectories)} trajectories from {path}")

        num_to_sample = int(len(trajectories) * weight * len(demo_paths))
        sampled_indices = np.random.choice(len(trajectories), size=num_to_sample, replace=True)
        sampled_trajs = [trajectories[i] for i in sampled_indices]

        combined_trajectories.extend(sampled_trajs)

    # Compute stats from the actual combined data
    combined_stats = compute_stats_from_trajectories(combined_trajectories)

    metadata = {
        'source_paths': demo_paths,
        'weights': weights,
    }
    save_demonstrations(combined_trajectories, combined_stats, output_path, metadata)

    return combined_trajectories, combined_stats


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract or augment expert demonstrations.",
    )
    subparsers = parser.add_subparsers(dest='command')

    aug_parser = subparsers.add_parser(
        'augment',
        help='Augment existing .pkl demos with cumulative discounted SVO returns.',
    )
    aug_parser.add_argument('--input', type=str, required=True,
                            help='Path to existing demonstrations .pkl file.')
    aug_parser.add_argument('--output', type=str, default=None,
                            help='Output path. Defaults to <input>_cumulative.pkl')
    aug_parser.add_argument('--gamma-svo', type=float, default=0.99,
                            help='Discount factor for SVO returns (default: 0.99).')

    args = parser.parse_args()

    if args.command == 'augment':
        trajectories, stats, metadata = load_demonstrations(args.input)

        sample_len = len(trajectories[0][0]) if trajectories and trajectories[0] else 0
        if sample_len < 8:
            print(f"ERROR: Transitions have {sample_len} elements, need >= 8.")
            sys.exit(1)

        print(f"Loaded {len(trajectories)} trajectories from {args.input}")
        print(f"Computing cumulative SVO returns with γ_svo={args.gamma_svo}...")

        augmented = compute_cumulative_svo_returns(trajectories, args.gamma_svo)

        metadata = metadata or {}
        metadata['gamma_svo'] = args.gamma_svo
        metadata['cumulative_svo'] = True

        output_path = args.output or args.input.replace('.pkl', '_cumulative.pkl')
        save_demonstrations(augmented, stats, output_path, metadata)

        sample = augmented[0][0]
        print(f"\nVerification: tuple length = {len(sample)}")
        print(f"  r_self={sample[6]:.4f}, r_global={sample[7]:.4f}")
        print(f"  G_self={sample[8]:.4f}, G_global={sample[9]:.4f}")
    else:
        print("Usage: python extract_demonstrations.py augment --input <path> --gamma-svo 0.99")