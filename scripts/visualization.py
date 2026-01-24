import argparse
import time
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO, DQN
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
        model = DQN.load(args.model_path, env=env)
    except FileNotFoundError:
        print(f"Couldn't find model file at {args.model_path}")

    # Run simulation
    obs, info = env.reset(seed=args.seed)
    print("Starting Visualization...")

    pygame.font.init()
    font = pygame.font.Font(None, 28)

    try:
        for _ in range(args.total_timesteps):
            action, _states = model.predict(obs, deterministic=True)

            ego_vehicle = env.unwrapped.vehicle
            neighbours = env.unwrapped.road.close_vehicles_to(
                vehicle=ego_vehicle,
                distance=50.0,
                count=env.neighbour_count,
                sort=True
            )


            svo_neighbors = [v for v in neighbours if v is not ego_vehicle]

            obs, reward, terminated, truncated, info = env.step(action)

            for vehicle in env.unwrapped.road.vehicles:
                if vehicle is ego_vehicle:
                    vehicle.color = (50, 200, 0)  # Green for ego (default)
                elif vehicle in svo_neighbors:
                    vehicle.color = (255, 165, 0)  # Orange for SVO neighbors
                else:
                    vehicle.color = (100, 100, 200)  # Blue for others

            env.render()

            viewer = env.unwrapped.viewer
            if viewer is not None:
                screen = viewer.screen
                speed_ms = env.unwrapped.vehicle.speed
                speed_km = speed_ms * 3.6

                text_str = f"Speed: {speed_km} km/sh"

                if speed_km > 25:
                    text_color = (50, 255, 50)  # Green
                elif speed_km < 10:
                    text_color = (255, 50, 50)  # Red
                else:
                    text_color = (255, 255, 255)  # White

                text_surface = font.render(text_str, True, text_color)
                padding = 5
                box_rect = text_surface.get_rect(topleft=(10, 10))
                box_rect.inflate_ip(padding*2, padding*2)
                pygame.draw.rect(screen, (30, 30, 30), box_rect, border_radius=5)

                screen.blit(text_surface, (15, 15))
                pygame.display.flip()

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