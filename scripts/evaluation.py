import argparse
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import DQN, PPO
from highway_env.envs import HighwayEnv
from rich.console import Console
from rich.table import Table

from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper

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


def run_seed(model, model_type, env, svo_alpha_rad, num_episodes, seed):
    """
    Run evaluation for a single seed. Returns a dict of per-seed aggregated metrics.
    """
    results = {
        "rewards_total": [],
        "rewards_self": [],
        "rewards_global": [],
        "speeds": [],
        "collisions": 0,
        "successes": 0,
        "overtakes": [],
        "lane_changes": [],
        "lane_preference": [],
        "tailgating_time": [],
        "brake_checks": [],
    }

    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)

        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_self = 0.0
        ep_global = 0.0
        ep_speeds = []

        ep_overtakes = 0
        ep_lane_changes = 0
        ep_lane_position = []
        ep_tailgating_steps = 0
        ep_brake_checks = 0
        total_steps = 0

        ego_vehicle = env.unwrapped.vehicle
        ego_length = ego_vehicle.LENGTH
        last_speed = ego_vehicle.speed

        vehicles_in_front = set()
        ego_x = ego_vehicle.position[0]

        for v in env.unwrapped.road.vehicles:
            if v is ego_vehicle:
                continue
            if v.position[0] > ego_x:
                vehicles_in_front.add(v)

        last_lane = ego_vehicle.lane_index[2]

        while not (terminated or truncated):
            action = predict_action(model, model_type, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_self += info.get("rewards/component_self", 0.0)
            ep_global += info.get("rewards/component_global", 0.0)

            speed_km = ego_vehicle.speed * 3.6
            ep_speeds.append(speed_km)

            total_steps += 1

            # Tailgate tracking
            lead_vehicle = None
            min_distance = float('inf')

            for v in env.unwrapped.road.vehicles:
                if v is ego_vehicle:
                    continue
                if (v.lane_index[2] == ego_vehicle.lane_index[2] and
                        v.position[0] > ego_vehicle.position[0]):
                    distance = v.position[0] - ego_vehicle.position[0]
                    if distance < min_distance:
                        min_distance = distance
                        lead_vehicle = v

            if lead_vehicle is not None:
                distance = lead_vehicle.position[0] - ego_vehicle.position[0] - ego_length
                time_headway = distance / (ego_vehicle.speed + 1e-6)
                if time_headway < 1.0:
                    ep_tailgating_steps += 1

            # Brake check tracking
            dt = 1.0 / env.unwrapped.config["simulation_frequency"]
            accel = (ego_vehicle.speed - last_speed) / dt

            rear_vehicle = None
            min_rear_distance = float('inf')

            for v in env.unwrapped.road.vehicles:
                if v is ego_vehicle:
                    continue
                if (v.lane_index[2] == ego_vehicle.lane_index[2] and
                        v.position[0] < ego_vehicle.position[0]):
                    distance = ego_vehicle.position[0] - v.position[0]
                    if distance < min_rear_distance:
                        min_rear_distance = distance
                        rear_vehicle = v

            if rear_vehicle is not None and min_rear_distance < 30:
                if accel < -3.0:
                    ep_brake_checks += 1

            last_speed = ego_vehicle.speed

            # Lane tracking
            current_lane = ego_vehicle.lane_index[2]
            num_lanes = len(env.unwrapped.road.network.graph["0"]["1"])
            norm_lane = current_lane / max(num_lanes - 1, 1)
            ep_lane_position.append(norm_lane)

            if current_lane != last_lane:
                ep_lane_changes += 1
                last_lane = current_lane

            # Overtake tracking
            ego_x = ego_vehicle.position[0]
            margin = ego_length * 0.5

            passed = []
            for v in vehicles_in_front:
                if v.position[0] < ego_x - margin:
                    ep_overtakes += 1
                    passed.append(v)

            for v in passed:
                vehicles_in_front.remove(v)

            for v in env.unwrapped.road.vehicles:
                if v is ego_vehicle:
                    continue
                if v.position[0] > ego_x + margin and v not in vehicles_in_front:
                    vehicles_in_front.add(v)

        # Episode aggregation
        results["rewards_total"].append(ep_reward)
        results["rewards_self"].append(ep_self)
        results["rewards_global"].append(ep_global)
        results["speeds"].append(np.mean(ep_speeds) if ep_speeds else 0.0)
        results["overtakes"].append(ep_overtakes)
        results["lane_changes"].append(ep_lane_changes)
        results["lane_preference"].append(
            np.mean(ep_lane_position) if ep_lane_position else 0.0
        )
        results["tailgating_time"].append(
            100 * ep_tailgating_steps / total_steps if total_steps > 0 else 0.0
        )
        results["brake_checks"].append(ep_brake_checks)

        if ego_vehicle.crashed:
            results["collisions"] += 1
        else:
            results["successes"] += 1

    # Compute seed-level aggregates
    n = num_episodes
    return {
        "collision_rate": 100 * results["collisions"] / n,
        "success_rate": 100 * results["successes"] / n,
        "avg_speed": np.mean(results["speeds"]),
        "overtakes": np.mean(results["overtakes"]),
        "lane_changes": np.mean(results["lane_changes"]),
        "lane_preference": np.mean(results["lane_preference"]),
        "tailgating_time": np.mean(results["tailgating_time"]),
        "brake_checks": np.mean(results["brake_checks"]),
        "reward_total": np.mean(results["rewards_total"]),
        "r_self": np.mean(results["rewards_self"]),
        "r_global": np.mean(results["rewards_global"]),
        "global_self_ratio": (
            np.mean(results["rewards_global"]) / np.mean(results["rewards_self"])
            if np.mean(results["rewards_self"]) != 0 else 0.0
        ),
    }


def evaluate(args):
    config = ENV_CONFIG.copy()
    config.update({
        "render_agent": False,
        "offscreen_rendering": True,
    })

    env = gym.make(config["id"], config=config)
    env.unwrapped.configure(config)

    svo_alpha_rad = np.deg2rad(args.svo_angle)
    env = SVOPureWrapper(env, svo_alpha=svo_alpha_rad)

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
            f"speed={seed_result['avg_speed']:.1f}km/h  "
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
    seed_table = Table(title=f"Per-Seed Results [{model_label}]")
    seed_table.add_column("Seed", style="cyan")
    seed_table.add_column("Collision %", style="magenta")
    seed_table.add_column("Speed (km/h)", style="white")
    seed_table.add_column("Reward", style="white")
    seed_table.add_column("R_self", style="white")
    seed_table.add_column("R_global", style="white")
    seed_table.add_column("Tailgating %", style="white")
    seed_table.add_column("Overtakes", style="white")

    for seed, r in zip(seeds, all_seed_results):
        seed_table.add_row(
            str(seed),
            f"{r['collision_rate']:.1f}",
            f"{r['avg_speed']:.1f}",
            f"{r['reward_total']:.1f}",
            f"{r['r_self']:.1f}",
            f"{r['r_global']:.1f}",
            f"{r['tailgating_time']:.1f}",
            f"{r['overtakes']:.1f}",
        )

    console.print("\n")
    console.print(seed_table)

    # Summary table
    summary = Table(title=f"Aggregated: {len(seeds)} seeds × {args.episodes} ep [{model_label}]")
    summary.add_column("Metric", style="cyan", no_wrap=True)
    summary.add_column("Mean", style="magenta")
    summary.add_column("± Std (across seeds)", style="green")

    summary.add_section()
    summary.add_row("Collision Rate",
                    f"{agg['collision_rate']['mean']:.1f}%",
                    f"± {agg['collision_rate']['std']:.1f}%")
    summary.add_row("Success Rate",
                    f"{agg['success_rate']['mean']:.1f}%",
                    f"± {agg['success_rate']['std']:.1f}%")
    summary.add_row("Avg Speed",
                    f"{agg['avg_speed']['mean']:.1f} km/h",
                    f"± {agg['avg_speed']['std']:.1f} km/h")

    summary.add_section()
    summary.add_row("Overtakes / Episode",
                    f"{agg['overtakes']['mean']:.2f}",
                    f"± {agg['overtakes']['std']:.2f}")
    summary.add_row("Lane Changes / Episode",
                    f"{agg['lane_changes']['mean']:.2f}",
                    f"± {agg['lane_changes']['std']:.2f}")
    summary.add_row("Lane Preference (0=L, 1=R)",
                    f"{agg['lane_preference']['mean']:.2f}",
                    f"± {agg['lane_preference']['std']:.2f}")
    summary.add_row("Tailgating Time (%)",
                    f"{agg['tailgating_time']['mean']:.1f}%",
                    f"± {agg['tailgating_time']['std']:.1f}%")
    summary.add_row("Brake Checks / Episode",
                    f"{agg['brake_checks']['mean']:.2f}",
                    f"± {agg['brake_checks']['std']:.2f}")

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