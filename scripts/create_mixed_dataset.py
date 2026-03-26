"""
Create mixed demonstration datasets.

Each source can be either:
  - A pre-extracted .pkl file
  - A DQN model .zip file (will be extracted on the fly)

For models, you must also provide the SVO angle (in degrees) that the
environment wrapper should use during extraction.

Usage examples:

  # From two existing .pkl files (50/50 split):
  python -m scripts.create_mixed_dataset \
      --sources altruistic.pkl egoistic.pkl \
      --weights 0.5 0.5 \
      --output ./expert_demonstrations/mixed_50_50.pkl

  # From two models (extracts then combines):
  python -m scripts.create_mixed_dataset \
      --sources data/experts/ego/final_model.zip data/experts/alt/final_model.zip \
      --svo-angles 0 90 \
      --weights 0.5 0.5 \
      --output ./expert_demonstrations/mixed_50_50.pkl

  # Mix of .pkl and model:
  python -m scripts.create_mixed_dataset \
      --sources altruistic.pkl data/experts/ego/final_model.zip \
      --svo-angles _ 0 \
      --weights 0.5 0.5 \
      --output ./expert_demonstrations/mixed.pkl
"""

import argparse
import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN

from scripts.extract_demonstrations import (
    extract_demonstrations_from_agent,
    save_demonstrations,
    load_demonstrations,
    combine_demonstrations,
    compute_stats_from_trajectories,
)
# from src.envs.svo_pure_wrapper import SVOPureWrapper
# from configs.env_config import ENV_CONFIG

from configs.intersection_config_new import INTERSECTION_CONFIG as ENV_CONFIG
from src.envs.intersection_yielding_wrapper import SVOYieldingWrapper as SVOPureWrapper


def create_env(svo_angle_deg: float, global_aggregation: str = 'mean'):
    """Create highway env with SVOPureWrapper at the given angle (degrees)."""
    svo_rad = np.deg2rad(svo_angle_deg)
    env = gym.make(ENV_CONFIG['id'])
    env.unwrapped.config.update(ENV_CONFIG)
    env = SVOPureWrapper(env, svo_alpha=svo_rad, lamb=1.0,
                         global_aggregation=global_aggregation)
    return env


def resolve_source(
    source_path: str,
    svo_angle_deg: float | None,
    num_episodes: int,
    save_dir: str,
    deterministic: bool = True,
    global_aggregation: str = 'mean',
) -> str:
    """
    Given a source path, return a path to a .pkl file with demonstrations.

    If source_path is already a .pkl, return it directly.
    If it's a .zip model, extract demonstrations and save a .pkl, then return
    the path to that .pkl.
    """
    if source_path.endswith('.pkl'):
        print(f"Using existing demonstrations: {source_path}")
        return source_path

    if source_path.endswith('.zip'):
        if svo_angle_deg is None:
            raise ValueError(
                f"Model source '{source_path}' requires an --svo-angles entry. "
                f"Use the degree value for the env wrapper (e.g. 0, 45, 90)."
            )

        # Derive a name from the model path
        model_name = os.path.splitext(os.path.basename(source_path))[0]
        agg_tag = f"_{global_aggregation}" if global_aggregation != 'mean' else ""
        pkl_name = f"{model_name}_svo{svo_angle_deg:.0f}deg{agg_tag}_demonstrations.pkl"
        pkl_path = os.path.join(save_dir, pkl_name)

        print(f"\n{'='*60}")
        print(f"Extracting from model: {source_path}")
        print(f"SVO angle: {svo_angle_deg}°")
        print(f"Global aggregation: {global_aggregation}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}")

        env = create_env(svo_angle_deg, global_aggregation=global_aggregation)
        agent = DQN.load(source_path)

        trajectories, stats = extract_demonstrations_from_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes,
            deterministic=deterministic,
            verbose=True,
        )

        metadata = {
            'agent_path': source_path,
            'svo_angle_deg': svo_angle_deg,
            'deterministic': deterministic,
            'global_aggregation': global_aggregation,
        }
        save_demonstrations(trajectories, stats, pkl_path, metadata)
        env.close()

        return pkl_path

    raise ValueError(
        f"Unsupported source format: '{source_path}'. "
        f"Expected .pkl or .zip"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Create mixed demonstration datasets',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--sources', type=str, nargs='+', required=True,
        help='Paths to .pkl demo files or .zip DQN models.',
    )
    parser.add_argument(
        '--svo-angles', type=str, nargs='*', default=None,
        help='SVO angle in degrees per source (required for .zip models).\n'
             'Use "_" or "none" to skip for .pkl sources.\n'
             'Example: --svo-angles 0 90\n'
             'Example: --svo-angles _ 90   (first source is .pkl, second is .zip)',
    )
    parser.add_argument(
        '--weights', type=float, nargs='+', default=None,
        help='Sampling weights per source (must sum to 1). Defaults to uniform.',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for the combined .pkl file.',
    )
    parser.add_argument(
        '--num-episodes', type=int, default=200,
        help='Episodes to extract per model source (ignored for .pkl sources).',
    )
    parser.add_argument(
        '--save-dir', type=str, default='./expert_demonstrations',
        help='Directory to save intermediate extracted .pkl files.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed.',
    )
    parser.add_argument(
        '--global-aggregation', type=str, default='mean',
        choices=['mean', 'min'],
        help='How to aggregate neighbor rewards into r_global.\n'
             '  mean (default): Average of all neighbor rewards.\n'
             '  min: Minimum neighbor reward — captures worst-affected vehicle.',
    )

    parser.add_argument(
        '--max-transitions-per-source', type=int, default=None,
        help='Cap transitions per source for balanced datasets.\\n'
             'E.g. --max-transitions-per-source 500 gives 1500 total for 3 sources.',
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    n_sources = len(args.sources)

    # Parse SVO angles
    svo_angles = [None] * n_sources
    if args.svo_angles is not None:
        if len(args.svo_angles) != n_sources:
            raise ValueError(
                f"Got {len(args.svo_angles)} --svo-angles but {n_sources} --sources. "
                f"Must match."
            )
        for i, val in enumerate(args.svo_angles):
            if val.lower() in ('_', 'none', 'n/a'):
                svo_angles[i] = None
            else:
                svo_angles[i] = float(val)

    # Check that all .zip sources have an SVO angle
    for src, angle in zip(args.sources, svo_angles):
        if src.endswith('.zip') and angle is None:
            raise ValueError(
                f"Model source '{src}' needs an SVO angle. "
                f"Provide it via --svo-angles."
            )

    # Resolve all sources to .pkl paths
    print(f"\n{'='*60}")
    print(f"Resolving {n_sources} sources...")
    print(f"{'='*60}\n")

    pkl_paths = []
    for src, angle in zip(args.sources, svo_angles):
        pkl_path = resolve_source(
            source_path=src,
            svo_angle_deg=angle,
            num_episodes=args.num_episodes,
            save_dir=args.save_dir,
            global_aggregation=args.global_aggregation,
        )
        pkl_paths.append(pkl_path)

    # Combine
    print(f"\n{'='*60}")
    print(f"Combining {n_sources} sources...")
    print(f"{'='*60}\n")

    combined_trajs, combined_stats = combine_demonstrations(
        demo_paths=pkl_paths,
        output_path=args.output,
        weights=args.weights,
    )

    # ── Balance transitions per source ──
    if args.max_transitions_per_source:
        cap = args.max_transitions_per_source
        print(f"\\nBalancing to max {cap} transitions per source...")

        # Re-load each source individually and truncate
        balanced_trajs = []
        for pkl_path in pkl_paths:
            trajs, _, _ = load_demonstrations(pkl_path)
            count = 0
            for traj in trajs:
                if count >= cap:
                    break
                remaining = cap - count
                if len(traj) <= remaining:
                    balanced_trajs.append(traj)
                    count += len(traj)
                else:
                    balanced_trajs.append(traj[:remaining])
                    count += remaining
            print(f"  {pkl_path}: {count} transitions")

        # Re-save with balanced data
        combined_stats = compute_stats_from_trajectories(balanced_trajs)
        metadata = {
            'source_paths': pkl_paths,
            'weights': args.weights or [1.0 / len(pkl_paths)] * len(pkl_paths),
            'mode_labels': [],  # Will be filled below
        }

        # Assign mode labels from source paths
        for pkl_path in pkl_paths:
            trajs_source, _, _ = load_demonstrations(pkl_path)
            count = 0
            for traj in trajs_source:
                if count >= cap:
                    break
                remaining = cap - count
                t_len = min(len(traj), remaining)
                count += t_len
            # Extract mode name from path
            import re
            name_match = re.search(r'svo(\d+)deg', pkl_path)
            mode_name = f"svo_{name_match.group(1)}" if name_match else pkl_path
            metadata['mode_labels'].extend([mode_name] * count)

        save_demonstrations(balanced_trajs, combined_stats, args.output, metadata)
        combined_trajs = balanced_trajs
        print(f"  Total: {sum(len(t) for t in balanced_trajs)} transitions")

    print(f"\n{'='*60}")
    print("Mixed Dataset Created")
    print(f"{'='*60}")
    print(f"Total episodes:     {combined_stats['num_episodes']}")
    print(f"Total transitions:  {combined_stats['total_transitions']}")
    print(f"Mean reward:        {combined_stats['mean_reward']:.3f} ± {combined_stats['std_reward']:.3f}")
    print(f"Mean length:        {combined_stats['mean_length']:.1f} ± {combined_stats['std_length']:.1f}")
    print(f"Collision rate:     {combined_stats['collision_rate']:.2%}")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    main()