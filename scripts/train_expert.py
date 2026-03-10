import argparse
import os
import numpy as np
import gymnasium as gym
import wandb
import torch
from highway_env.envs import HighwayEnv

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback


from configs.intersection_config import INTERSECTION_CONFIG as ENV_CONFIG
from src.envs.svo_intersection_wrapper import SVOIntersectionWrapper as SVOPureWrapper

def make_env(rank: int, svo_alpha: float, seed: int = 42):
    """
    Utility function for multiprocessed env.
    :param rank: Index for sub-process
    :param svo_alpha: SVO angle in radians
    :param seed: Global seed
    :return: Gym environment
    """

    def _init():
        env = gym.make(ENV_CONFIG["id"], render_mode=None)
        env.unwrapped.configure(ENV_CONFIG)
        env = SVOPureWrapper(env, svo_alpha=svo_alpha)
        env.reset(seed=seed + rank)
        return env

    return _init

def train(args):
    # Convert SVO angle from degrees to radians
    svo_alpha_rad = np.deg2rad(args.svo_angle)
    run_name = f"expert_svo_intersection_{args.svo_angle}deg_seed{args.seed}"

    print(f"Starting Training")
    print(f"Expert: {run_name}")
    print(f"SVO Angle {args.svo_angle} degrees ({svo_alpha_rad:.2f} radians)")

    # WandB Initialization
    wandb.init(
        project = args.project_name,
        name = run_name,
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": args.total_timesteps,
            "svo_angle_deg": args.svo_angle,
            "env_config": ENV_CONFIG,
            "algorithm": "PPO",
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Create vectorized environment
    num_cpu = 12
    env = SubprocVecEnv([
        make_env(i, svo_alpha_rad, args.seed) for i in range(num_cpu)
    ])
    env = VecMonitor(env, filename=None)

    eval_env = SubprocVecEnv([
        make_env(99, svo_alpha_rad, args.seed + 10000)
    ])
    eval_env = VecMonitor(eval_env, filename=None)

    # Save path
    save_path = os.path.join("data", "experts", run_name)
    best_model_path = os.path.join(save_path, "best_model")
    os.makedirs(save_path, exist_ok=True)

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.95,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=500,
                verbose=1,
                tensorboard_log="highway_dqn/",
                device="cuda")



    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_cpu,
        save_path=save_path,
        name_prefix="ckpt",
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run_name}",
        verbose=2,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=save_path,
        eval_freq=50000 // num_cpu,
        deterministic=True,
        render=False,
        n_eval_episodes=100,
        verbose=1,
    )

    # Train Agent
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, wandb_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")

    # Final Save
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)

    wandb.save(f"{final_model_path}.zip")

    wandb.save(os.path.join(best_model_path, "best_model.zip"))

    print(f"Training completed and model saved to {final_model_path}.zip.")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVO Experts with PPO and WandB")

    # SVO angle in degrees
    parser.add_argument("--svo_angle", type=float, required=True, help="SVO angle in degrees")

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=300_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--project_name", type=str, default="svo-irl-thesis", help="WandB project name")

    args = parser.parse_args()
    train(args)
