import argparse
import os
import numpy as np
import gymnasium as gym
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imitation_logger

from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper


def make_env_fn(env_id, config, svo_angle=None):
    """
    Creates an environment.
    - If svo_angle is provided, it wraps the env (for Expert generation).
    - If svo_angle is None, it returns the raw env (for Student learning).
    """

    def _init():
        env = gym.make(env_id, config=config)
        env.unwrapped.configure(config)

        if svo_angle is not None:
            svo_rad = np.deg2rad(svo_angle)
            env = SVOPureWrapper(env, svo_alpha=svo_rad)

        # Add Monitor wrapper - critical for imitation's rollout
        env = Monitor(env)
        return env

    return _init


def train_mixed_airl(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="airl_mode_collapse",
        sync_tensorboard=True,
        config=vars(args),
        name=f"AIRL_Mixed_0_90_{args.seed}_optimized",
        monitor_gym=True,
        save_code=True,
    )

    print(f"\n=== Experiment 1: Mode Collapse (50% SVO 0° / 50% SVO 90°) ===")
    print(f"=== OPTIMIZED VERSION - Faster Training ===\n")

    # 1. Setup Config
    config = ENV_CONFIG.copy()

    wandb.config.update({"env_config": config})

    # 2. Load Expert 0 and Generate Demonstrations
    print(f"\n--> Loading Egoistic Expert (SVO 0°) from {args.path_1}")
    model_0 = DQN.load(args.path_1)  # Load without env parameter

    # Create environment for rollout generation
    env_0 = DummyVecEnv([make_env_fn(config["id"], config, svo_angle=0.0)])

    print(f"    Generating {args.n_demos} egoistic trajectories...")
    demos_0 = rollout.rollout(
        model_0,
        env_0,
        rollout.make_sample_until(min_episodes=args.n_demos),
        rng=np.random.default_rng(args.seed),
        unwrap=False,  # Don't unwrap - avoids needing rollout stats
    )

    env_0.close()

    # 3. Load Expert 90 and Generate Demonstrations
    print(f"\n--> Loading Altruistic Expert (SVO 90°) from {args.path_2}")
    model_90 = DQN.load(args.path_2)  # Load without env parameter

    env_90 = DummyVecEnv([make_env_fn(config["id"], config, svo_angle=90.0)])

    print(f"    Generating {args.n_demos} altruistic trajectories...")
    demos_90 = rollout.rollout(
        model_90,
        env_90,
        rollout.make_sample_until(min_episodes=args.n_demos),
        rng=np.random.default_rng(args.seed + 1),
        unwrap=False,  # Don't unwrap - avoids needing rollout stats
    )

    env_90.close()

    # 4. Mix Demonstrations (The "Bimodal" Dataset)
    rng = np.random.default_rng(args.seed + 2)
    mixed_demos = list(demos_0) + list(demos_90)
    rng.shuffle(mixed_demos)  # Shuffle to break sequential bias

    print(f"\n--> Created Mixed Dataset: {len(mixed_demos)} total trajectories")
    print(f"    (0° count: {len(demos_0)}, 90° count: {len(demos_90)})")

    # Log trajectory statistics
    total_transitions_0 = sum(len(traj) for traj in demos_0)
    total_transitions_90 = sum(len(traj) for traj in demos_90)
    print(f"    Total transitions: 0° = {total_transitions_0}, 90° = {total_transitions_90}")

    # 5. Setup Logging Paths for AIRL
    log_dir = f"{args.output_dir}/{run.id}"
    os.makedirs(log_dir, exist_ok=True)

    # Configure Imitation Library Logger
    # This ensures AIRL discriminator stats (disc_loss, etc.) go to Tensorboard
    custom_logger = imitation_logger.configure(log_dir, ["stdout", "tensorboard"])

    # 6. Setup AIRL Student Environment - OPTIMIZED
    # Use SubprocVecEnv for true CPU parallelism (2-3x faster than DummyVecEnv)
    # Increased to 24 envs for better throughput
    print(f"\n--> Creating {args.n_envs} parallel environments with SubprocVecEnv...")
    venv = SubprocVecEnv([make_env_fn(config["id"], config, svo_angle=None)
                          for _ in range(args.n_envs)])

    # 7. Initialize AIRL Components - OPTIMIZED
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # OPTIMIZED PPO: Tuned for i7-12700H (14 cores) + RTX 3070 Laptop (8GB VRAM)
    # - Reduced n_epochs from 10 to 5 (2x faster per update)
    # - Reduced n_steps from 2048 to 512 (faster rollout collection)
    # - Increased batch_size from 64 to 128 (better GPU utilization)
    print(f"\n--> Initializing PPO learner (optimized for RTX 3070)...")
    learner = PPO(
        env=venv,
        policy="MlpPolicy",
        batch_size=128,  # Increased for better GPU utilization
        ent_coef=0.01,  # Slight entropy to encourage exploration before collapse
        learning_rate=3e-4,
        n_epochs=5,  # Reduced from 10 (2x speedup)
        n_steps=512,  # Reduced from 2048 (faster rounds)
        verbose=2,  # Detailed output
        tensorboard_log=log_dir,
        device='cuda',  # RTX 3070
    )

    # OPTIMIZED AIRL: Reduced discriminator updates for faster training
    # - n_disc_updates_per_round: 2 instead of 4 (2x faster discriminator phase)
    print(f"--> Initializing AIRL trainer (optimized)...")
    airl_trainer = AIRL(
        demonstrations=mixed_demos,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=2,  # Reduced from 4 (2x speedup)
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=custom_logger,
    )

    # 8. Train
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"  Total timesteps:              {args.total_timesteps:,}")
    print(f"  Parallel environments:        {args.n_envs}")
    print(f"  PPO batch size:               128")
    print(f"  PPO n_epochs:                 5 (reduced from 10)")
    print(f"  PPO n_steps:                  512 (reduced from 2048)")
    print(f"  Discriminator updates/round:  2 (reduced from 4)")
    print(f"  Demo batch size:              1,024")
    print(f"  Device:                       CUDA (RTX 3070)")
    print(f"  CPU:                          i7-12700H ({args.n_envs} workers)")
    print(f"{'='*70}")
    print(f"  Estimated time: ~1.5-2.5 hours (vs 9 hours unoptimized)")
    print(f"{'='*70}\n")

    # Note: WandbCallback is not passed here because AIRL doesn't support it directly
    # TensorBoard logs will automatically sync to WandB via sync_tensorboard=True
    airl_trainer.train(total_timesteps=args.total_timesteps)

    # 9. Save Final Models
    save_path = os.path.join(args.output_dir, "airl_mode_collapse_mixed")
    os.makedirs(save_path, exist_ok=True)

    learner.save(os.path.join(save_path, "gen_policy"))
    torch.save(reward_net.state_dict(), os.path.join(save_path, "reward_net.pt"))

    print(f"\n--> Models saved to {save_path}")

    # 10. Upload final model to WandB as an artifact
    artifact = wandb.Artifact(f"airl_policy_{args.seed}", type="model")
    artifact.add_file(os.path.join(save_path, "gen_policy.zip"))
    run.log_artifact(artifact)

    print(f"\n--> Experiment Complete!")
    print("    Next Step: Evaluate this policy using evaluate.py to see if it")
    print("    behaves like SVO 0°, SVO 90°, or fails (crashes).")

    venv.close()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Expert Paths
    parser.add_argument("--path_1", type=str, required=True,
                        help="Path to SVO 0° (Egoistic) model.zip")
    parser.add_argument("--path_2", type=str, required=True,
                        help="Path to SVO 90° (Altruistic) model.zip")

    # Training Config - OPTIMIZED DEFAULTS
    parser.add_argument("--output_dir", type=str, default="runs/airl_baseline",
                        help="Save directory")
    parser.add_argument("--n_demos", type=int, default=30,
                        help="Demos per expert (Total = 2 * n_demos)")
    parser.add_argument("--total_timesteps", type=int, default=200_000,
                        help="Training steps (reduced from 500k for faster baseline)")
    parser.add_argument("--n_envs", type=int, default=12,
                        help="Number of parallel environments (optimized for i7-12700H)")
    parser.add_argument("--seed", type=int, default=42)

    # WandB Config
    parser.add_argument("--wandb_project", type=str, default="highway-airl-svo",
                        help="WandB Project Name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB Entity (User/Team)")

    args = parser.parse_args()
    train_mixed_airl(args)