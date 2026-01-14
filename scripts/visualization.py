import argparse
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from highway_env.envs import HighwayEnv

from configs.env_config import ENV_CONFIG
from src.envs.svo_wrapper import SVOWrapper

def visualize(args):
    config = ENV_CONFIG.copy()
    config.update({
        "render_agent": True,
        "show_trajectories": True,
        "offscreen_rendering": False,
    })

    env = gym.make(config["id"], render_mode="rgb_array", config=config)
    env.unwrapped.configure(config)
    env = SVOWrapper(env, svo_alpha=np.deg2rad(args.svo_angle))
    env.reset(seed=args.seed)


    # Load model
    try:
        model = PPO.load(args.model_path, env=env)
    except FileNotFoundError:
        print(f"Couldn't find model file at {args.model_path}")

    # Run simulation
    obs, info = env.reset(seed=args.seed)
    print("Starting Visualization...")

    try:
        for _ in range(args.total_timesteps):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            if terminated or truncated:
                print(f"Episode finished. SVO Reward: {info.get('rewards/svo_total', 0):.2f}")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained SVO agent")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--svo_angle", type=float, default=0.0, help="SVO angle used during training (default: 0)")
    parser.add_argument("--total_timesteps", type=int, default=1_000, help="Number of steps to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for traffic generation")

    args = parser.parse_args()
    visualize(args)