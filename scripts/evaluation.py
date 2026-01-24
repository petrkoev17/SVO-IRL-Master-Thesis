import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN, PPO
from highway_env.envs import HighwayEnv
from rich.console import Console
from rich.table import Table
from sympy.solvers.solvers import recast_to_symbols

from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper

def evaluate(args):
    # Env setup
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
        model = DQN.load(args.model_path, env=env)
    except FileNotFoundError:
        console.print(f"[red]Couldn't find model file at {args.model_path}[/red]")
        return

    console.print(f"Starting evaluation over {args.episodes} episodes...")

    # Metrics
    results = {
        "rewards_total": [],
        "rewards_self": [],
        "rewards_global": [],
        "speeds": [],
        "svo_episode": [],
        "svo_success": [],
        "svo_collision": [],
        "collisions": 0,
        "successes": 0,
        "overtakes": [],
        "lane_changes": [],
        "lane_preference": [],
        "tailgating_time": [],
        "brake_checks": [],
    }

    # Eval loop
    for i in range(args.episodes):
        obs, info = env.reset(seed=args.seed + i)

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
            action, _ = model.predict(obs, deterministic=True)
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

            # Find rear vehicle
            rear_vehicle = None
            min_rear_distance = float('inf')

            for v in env.unwrapped.road.vehicles:
                if v is ego_vehicle:
                    continue
                # Check if in same lane and behind
                if (v.lane_index[2] == ego_vehicle.lane_index[2] and
                        v.position[0] < ego_vehicle.position[0]):
                    distance = ego_vehicle.position[0] - v.position[0]
                    if distance < min_rear_distance:
                        min_rear_distance = distance
                        rear_vehicle = v

            # Check for brake checking: hard braking with car close behind
            if rear_vehicle is not None and min_rear_distance < 30:
                if accel < -3.0:  # Hard braking threshold
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
        if ep_self != 0.0 or ep_global != 0.0:
            svo_ep = np.degrees(np.arctan2(ep_global * np.cos(svo_alpha_rad), np.sin(svo_alpha_rad) * ep_self)) # * cos * sin
        else:
            svo_ep = 0.0

        results["rewards_total"].append(ep_reward)
        results["rewards_self"].append(ep_self)
        results["rewards_global"].append(ep_global)
        results["speeds"].append(np.mean(ep_speeds) if ep_speeds else 0.0)
        results["svo_episode"].append(svo_ep)
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
            results["svo_collision"].append(svo_ep)
        else:
            results["successes"] += 1
            results["svo_success"].append(svo_ep)

        print(f"   Episode {i + 1}/{args.episodes} complete.", end="\r")
    env.close()

    # Aggregate statistics
    n = args.episodes
    collision_rate = 100 * results["collisions"] / n
    success_rate = 100 * results["successes"] / n


    avg_self = np.mean(results["rewards_self"])
    avg_global = np.mean(results["rewards_global"])
    avg_svo_episode = np.mean(results["svo_episode"])

    delta = ((avg_svo_episode - args.svo_angle + 180) % 360) - 180

    # Build Table
    table = Table(title=f"Evaluation: {args.svo_angle}° Agent (Seed {args.seed})")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean", style="magenta")
    table.add_column("Std Dev", style="green")

    table.add_section()
    table.add_row("Collision Rate", f"{collision_rate:.1f}%", "-")
    table.add_row("Success Rate", f"{success_rate:.1f}%", "-")
    table.add_row("Avg Speed",
                  f"{np.mean(results['speeds']):.2f} km/h",
                  f"± {np.std(results['speeds']):.2f} km/h")

    table.add_section()
    table.add_row(
        "Overtakes / Episode",
        f"{np.mean(results['overtakes']):.2f}",
        f"± {np.std(results['overtakes']):.2f}",
    )
    table.add_row(
        "Lane Changes / Episode",
        f"{np.mean(results['lane_changes']):.2f}",
        f"± {np.std(results['lane_changes']):.2f}",
    )
    table.add_row(
        "Lane Preference (0=Left,1=Right)",
        f"{np.mean(results['lane_preference']):.2f}",
        f"± {np.std(results['lane_preference']):.2f}",
    )
    table.add_row(
        "Time Spent Tailgating (%)",
        f"{np.mean(results['tailgating_time']):.2f}%",
        f"± {np.std(results['tailgating_time']):.2f}%",
    )
    table.add_row(
        "Brake Check Events / Episode",
        f"{np.mean(results['brake_checks']):.2f}",
        f"± {np.std(results['brake_checks']):.2f}",
    )

    table.add_section()
    table.add_row(
        "Total SVO Reward",
        f"{np.mean(results['rewards_total']):.2f}",
        f"± {np.std(results['rewards_total']):.2f}",
    )
    table.add_row(
        "Self Utility (R_self)",
        f"{avg_self:.2f}",
        f"± {np.std(results['rewards_self']):.2f}",
    )
    table.add_row(
        "Global Utility (R_global)",
        f"{avg_global:.2f}",
        f"± {np.std(results['rewards_global']):.2f}",
    )

    table.add_section()
    table.add_row(
        "Empirical SVO (episode avg)",
        f"{avg_svo_episode:.2f}°",
        f"Δ = {delta:+.1f}°",
    )

    if results["svo_success"]:
        table.add_row(
            "SVO (success only)",
            f"{np.mean(results['svo_success']):.2f}°",
            f"± {np.std(results['svo_success']):.2f}°",
        )
    else:
        table.add_row("SVO (success only)", "n/a", "-")

    if results["svo_collision"]:
        table.add_row(
            "SVO (collision only)",
            f"{np.mean(results['svo_collision']):.2f}°",
            f"± {np.std(results['svo_collision']):.2f}°",
        )
    else:
        table.add_row("SVO (collision only)", "n/a", "-")

    console.print("\n")
    console.print(table)

    # Additional ratio information
    if avg_self != 0:
        console.print(
            f"[italic]Global/Self Ratio: {avg_global / avg_self:.3f}[/italic]"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--svo_angle", type=float, default=0.0, help="SVO angle used for reward calculation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    evaluate(args)