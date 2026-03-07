"""
Evaluation script for the Intersection environment.

Metrics:
  - Collision rate, success rate (arrived at destination)
  - Timeout rate (neither crashed nor arrived)
  - Average speed, episode length
  - Near-miss events (close proximity to other vehicles)
  - Hard braking events
  - Neighbor collisions (other vehicles crashing, potentially caused by ego)
  - SVO reward components (r_self, r_global)

Supports multi-seed evaluation with per-seed and aggregated tables.
"""

import argparse
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import DQN, PPO
from rich.console import Console
from rich.table import Table

from configs.intersection_config import INTERSECTION_CONFIG
from src.envs.svo_intersection_wrapper import SVOIntersectionWrapper
from src.algorithms.iq_learner import IQLearnTrainer


def load_model(model_path, env):
    """Load a model from either a .pt (IQ-Learn) or .zip (SB3) file."""
    if model_path.endswith('.pt'):
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n

        trainer = IQLearnTrainer(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        trainer.load(model_path)
        return trainer, 'iq_learn'
    else:
        try:
            model = DQN.load(model_path)
        except Exception:
            model = PPO.load(model_path)
        return model, 'sb3'


def predict_action(model, model_type, obs):
    """Get a deterministic action from the model."""
    if model_type == 'iq_learn':
        return model.select_action(obs, epsilon=0.0)
    else:
        action, _ = model.predict(obs, deterministic=True)
        return action


def has_arrived(env, vehicle, exit_distance=25):
    """Check if a vehicle has reached the intersection exit."""
    return (
        "il" in vehicle.lane_index[0]
        and "o" in vehicle.lane_index[1]
        and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
    )


def run_seed(model, model_type, env, svo_alpha_rad, num_episodes, seed):
    """
    Run evaluation for a single seed on the intersection environment.
    Returns a dict of per-seed aggregated metrics.
    """
    results = {
        "rewards_total": [],
        "rewards_self": [],
        "rewards_global": [],
        "speeds": [],
        "episode_lengths": [],
        "collisions": 0,
        "arrivals": 0,
        "timeouts": 0,
        "near_misses": [],
        "hard_brakes": [],
        "min_distances": [],
        "neighbor_collisions": [],
        "time_stopped": [],
    }

    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)

        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_self = 0.0
        ep_global = 0.0
        ep_speeds = []

        ep_near_misses = 0
        ep_hard_brakes = 0
        ep_min_distance = float('inf')
        ep_stopped_steps = 0
        total_steps = 0

        ego_vehicle = env.unwrapped.vehicle
        last_speed = ego_vehicle.speed

        # Track initial neighbor states for collision detection
        initial_neighbor_ids = set()
        for v in env.unwrapped.road.vehicles:
            if v is not ego_vehicle:
                initial_neighbor_ids.add(id(v))

        while not (terminated or truncated):
            action = predict_action(model, model_type, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_self += info.get("rewards/component_self", 0.0)
            ep_global += info.get("rewards/component_global", 0.0)

            speed_km = ego_vehicle.speed * 3.6
            ep_speeds.append(speed_km)

            total_steps += 1

            # Stopped time (speed < 0.5 m/s)
            if ego_vehicle.speed < 0.5:
                ep_stopped_steps += 1

            # Near-miss: any vehicle within 5m Euclidean distance
            for v in env.unwrapped.road.vehicles:
                if v is ego_vehicle:
                    continue
                dist = np.linalg.norm(ego_vehicle.position - v.position)
                ep_min_distance = min(ep_min_distance, dist)
                if dist < 5.0 and not ego_vehicle.crashed:
                    ep_near_misses += 1

            # Hard braking
            dt = 1.0 / env.unwrapped.config["simulation_frequency"]
            accel = (ego_vehicle.speed - last_speed) / dt
            if accel < -3.0:
                ep_hard_brakes += 1

            last_speed = ego_vehicle.speed

        # Determine episode outcome
        ego_arrived = has_arrived(env, ego_vehicle)
        ego_crashed = ego_vehicle.crashed

        # Count neighbor collisions
        neighbor_crashes = 0
        for v in env.unwrapped.road.vehicles:
            if v is not ego_vehicle and v.crashed:
                neighbor_crashes += 1

        # Episode aggregation
        results["rewards_total"].append(ep_reward)
        results["rewards_self"].append(ep_self)
        results["rewards_global"].append(ep_global)
        results["speeds"].append(np.mean(ep_speeds) if ep_speeds else 0.0)
        results["episode_lengths"].append(total_steps)
        results["near_misses"].append(ep_near_misses)
        results["hard_brakes"].append(ep_hard_brakes)
        results["min_distances"].append(ep_min_distance if ep_min_distance != float('inf') else 0.0)
        results["neighbor_collisions"].append(neighbor_crashes)
        results["time_stopped"].append(
            100 * ep_stopped_steps / total_steps if total_steps > 0 else 0.0
        )

        if ego_crashed:
            results["collisions"] += 1
        elif ego_arrived:
            results["arrivals"] += 1
        else:
            results["timeouts"] += 1

    # Compute seed-level aggregates
    n = num_episodes
    return {
        "collision_rate": 100 * results["collisions"] / n,
        "arrival_rate": 100 * results["arrivals"] / n,
        "timeout_rate": 100 * results["timeouts"] / n,
        "avg_speed": np.mean(results["speeds"]),
        "avg_length": np.mean(results["episode_lengths"]),
        "near_misses": np.mean(results["near_misses"]),
        "hard_brakes": np.mean(results["hard_brakes"]),
        "min_distance": np.mean(results["min_distances"]),
        "neighbor_collisions": np.mean(results["neighbor_collisions"]),
        "time_stopped": np.mean(results["time_stopped"]),
        "reward_total": np.mean(results["rewards_total"]),
        "r_self": np.mean(results["rewards_self"]),
        "r_global": np.mean(results["rewards_global"]),
        "global_self_ratio": (
            np.mean(results["rewards_global"]) / np.mean(results["rewards_self"])
            if np.mean(results["rewards_self"]) != 0 else 0.0
        ),
    }


def evaluate(args):
    config = INTERSECTION_CONFIG.copy()
    config.update({
        "render_agent": False,
        "offscreen_rendering": True,
    })

    env = gym.make(config["id"], config=config)
    env.unwrapped.configure(config)

    svo_alpha_rad = np.deg2rad(args.svo_angle)
    env = SVOIntersectionWrapper(env, svo_alpha=svo_alpha_rad)

    console = Console()

    # Load model
    try:
        model, model_type = load_model(args.model_path, env)
    except FileNotFoundError:
        console.print(f"[red]Couldn't find model file at {args.model_path}[/red]")
        return

    model_label = "IQ-Learn" if model_type == 'iq_learn' else "SB3"

    seeds = args.seeds
    console.print(f"Loaded {model_label} model from {args.model_path}")
    console.print(f"Evaluating over {len(seeds)} seeds × {args.episodes} episodes = {len(seeds) * args.episodes} total episodes")
    console.print(f"Seeds: {seeds}\n")

    # Run each seed
    all_seed_results = []
    for seed in seeds:
        console.print(f"  Running seed {seed}...", end=" ")
        seed_result = run_seed(model, model_type, env, svo_alpha_rad, args.episodes, seed)
        all_seed_results.append(seed_result)
        console.print(
            f"collision={seed_result['collision_rate']:.0f}%  "
            f"arrived={seed_result['arrival_rate']:.0f}%  "
            f"reward={seed_result['reward_total']:.1f}"
        )

    env.close()

    # Aggregate across seeds
    metrics = list(all_seed_results[0].keys())
    agg = {}
    for m in metrics:
        values = [r[m] for r in all_seed_results]
        agg[m] = {"mean": np.mean(values), "std": np.std(values), "values": values}

    # Per-seed table
    seed_table = Table(title=f"Per-Seed Results [{model_label}] (Intersection)")
    seed_table.add_column("Seed", style="cyan")
    seed_table.add_column("Collision %", style="magenta")
    seed_table.add_column("Arrived %", style="green")
    seed_table.add_column("Timeout %", style="yellow")
    seed_table.add_column("Speed (km/h)", style="white")
    seed_table.add_column("Reward", style="white")
    seed_table.add_column("R_self", style="white")
    seed_table.add_column("R_global", style="white")

    for seed, r in zip(seeds, all_seed_results):
        seed_table.add_row(
            str(seed),
            f"{r['collision_rate']:.1f}",
            f"{r['arrival_rate']:.1f}",
            f"{r['timeout_rate']:.1f}",
            f"{r['avg_speed']:.1f}",
            f"{r['reward_total']:.1f}",
            f"{r['r_self']:.1f}",
            f"{r['r_global']:.1f}",
        )

    console.print("\n")
    console.print(seed_table)

    # Summary table
    summary = Table(title=f"Aggregated: {len(seeds)} seeds × {args.episodes} ep [{model_label}] (Intersection)")
    summary.add_column("Metric", style="cyan", no_wrap=True)
    summary.add_column("Mean", style="magenta")
    summary.add_column("± Std (across seeds)", style="green")

    summary.add_section()
    summary.add_row("Collision Rate",
                    f"{agg['collision_rate']['mean']:.1f}%",
                    f"± {agg['collision_rate']['std']:.1f}%")
    summary.add_row("Arrival Rate",
                    f"{agg['arrival_rate']['mean']:.1f}%",
                    f"± {agg['arrival_rate']['std']:.1f}%")
    summary.add_row("Timeout Rate",
                    f"{agg['timeout_rate']['mean']:.1f}%",
                    f"± {agg['timeout_rate']['std']:.1f}%")
    summary.add_row("Avg Speed",
                    f"{agg['avg_speed']['mean']:.1f} km/h",
                    f"± {agg['avg_speed']['std']:.1f} km/h")
    summary.add_row("Avg Episode Length",
                    f"{agg['avg_length']['mean']:.1f} steps",
                    f"± {agg['avg_length']['std']:.1f}")

    summary.add_section()
    summary.add_row("Near Misses / Episode",
                    f"{agg['near_misses']['mean']:.2f}",
                    f"± {agg['near_misses']['std']:.2f}")
    summary.add_row("Hard Brakes / Episode",
                    f"{agg['hard_brakes']['mean']:.2f}",
                    f"± {agg['hard_brakes']['std']:.2f}")
    summary.add_row("Min Distance to Neighbor (m)",
                    f"{agg['min_distance']['mean']:.2f}",
                    f"± {agg['min_distance']['std']:.2f}")
    summary.add_row("Neighbor Collisions / Episode",
                    f"{agg['neighbor_collisions']['mean']:.2f}",
                    f"± {agg['neighbor_collisions']['std']:.2f}")
    summary.add_row("Time Stopped (%)",
                    f"{agg['time_stopped']['mean']:.1f}%",
                    f"± {agg['time_stopped']['std']:.1f}%")

    summary.add_section()
    summary.add_row("Total Reward",
                    f"{agg['reward_total']['mean']:.2f}",
                    f"± {agg['reward_total']['std']:.2f}")
    summary.add_row("Self Utility (R_self)",
                    f"{agg['r_self']['mean']:.2f}",
                    f"± {agg['r_self']['std']:.2f}")
    summary.add_row("Global Utility (R_global)",
                    f"{agg['r_global']['mean']:.2f}",
                    f"± {agg['r_global']['std']:.2f}")
    summary.add_row("Global/Self Ratio",
                    f"{agg['global_self_ratio']['mean']:.3f}",
                    f"± {agg['global_self_ratio']['std']:.3f}")

    console.print("\n")
    console.print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (.pt for IQ-Learn, .zip for SB3)")
    parser.add_argument("--svo_angle", type=float, default=0.0,
                        help="SVO angle used for reward calculation")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 200, 300, 400],
                        help="Evaluation seeds (default: 42 100 200 300 400)")

    args = parser.parse_args()
    evaluate(args)