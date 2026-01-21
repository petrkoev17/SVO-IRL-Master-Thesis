import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from highway_env.envs import HighwayEnv
from rich.console import Console
from rich.table import Table

from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper

def evaluate(args):
    config = ENV_CONFIG.copy()
    config.update({
        "render_agent": False,
        "offscreen_rendering": True,  # faster eval
    })

    env = gym.make(config["id"], config=config)
    env.unwrapped.configure(config)

    svo_alpha = np.deg2rad(args.svo_angle)
    env = SVOPureWrapper(env, svo_alpha=svo_alpha)

    console = Console()

    try:
        model = DQN.load(args.model_path, env=env)
    except FileNotFoundError:
        console.print(f"[red]Couldn't find model file at {args.model_path}[/red]")
        return

    console.print(f"🚀 Starting evaluation over {args.episodes} episodes...")

    results = {
        "rewards_total": [],
        "rewards_self": [],
        "rewards_global": [],
        "speeds": [],
        "collisions": 0,
        "successes": 0,
    }

    for i in range(args.episodes):
        obs, info = env.reset(seed=args.seed + i)

        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_self = 0.0
        ep_global = 0.0
        ep_speeds = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_self += info["rewards/component_self"]
            ep_global += info["rewards/component_global"]

            speed_km = env.unwrapped.vehicle.speed * 3.6
            ep_speeds.append(speed_km)

        results["rewards_total"].append(ep_reward)
        results["rewards_self"].append(ep_self)
        results["rewards_global"].append(ep_global)
        results["speeds"].append(np.mean(ep_speeds) if ep_speeds else 0.0)

        if env.unwrapped.vehicle.crashed:
            results["collisions"] += 1
        else:
            results["successes"] += 1  # finished episode without crashing

        print(f"   Episode {i + 1}/{args.episodes} complete.", end="\r")

    n = args.episodes
    collision_rate = 100 * results["collisions"] / n
    success_rate = 100 * results["successes"] / n
    avg_speed = np.mean(results["speeds"])

    table = Table(title=f"Evaluation: {args.svo_angle}° Agent (Seed {args.seed})")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean", style="magenta")
    table.add_column("Std Dev", style="green")

    table.add_section()
    table.add_row("Collision Rate", f"{collision_rate:.1f}%", "-")
    table.add_row("Success Rate", f"{success_rate:.1f}%", "-")
    table.add_row("Avg Speed", f"{avg_speed:.2f} km/h",
                  f"± {np.std(results['speeds']):.2f}")

    table.add_section()
    table.add_row("Total SVO Reward",
                  f"{np.mean(results['rewards_total']):.2f}",
                  f"± {np.std(results['rewards_total']):.2f}")
    table.add_row("Selfish Utility (R_self)",
                  f"{np.mean(results['rewards_self']):.2f}",
                  f"± {np.std(results['rewards_self']):.2f}")
    table.add_row("Social Utility (R_global)",
                  f"{np.mean(results['rewards_global']):.2f}",
                  f"± {np.std(results['rewards_global']):.2f}")

    console.print("\n")
    console.print(table)

    avg_self = np.mean(results["rewards_self"])
    avg_social = np.mean(results["rewards_global"])
    if avg_self != 0:
        console.print(f"[italic]Social/Self Ratio: {avg_social / avg_self:.3f}[/italic]")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--svo_angle", type=float, default=0.0, help="SVO angle used for reward calculation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    evaluate(args)