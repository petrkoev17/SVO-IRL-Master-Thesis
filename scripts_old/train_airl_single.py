import os
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import wandb

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imitation_logger
from imitation.rewards.reward_nets import RewardNet

from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper


class SurvivalBonusWrapper(RewardNet):
    """Adds survival bonus to prevent early termination."""

    def __init__(self, base_reward_net, bonus=1.0, floor=-0.5):
        super().__init__(
            observation_space=base_reward_net.observation_space,
            action_space=base_reward_net.action_space,
        )
        self.base_reward_net = base_reward_net
        self.bonus = bonus
        self.floor = floor

    def forward(self, obs, acts, next_obs, dones):
        raw_rew = self.base_reward_net(obs, acts, next_obs, dones)
        return torch.clamp(raw_rew + self.bonus, min=self.floor)

    def predict_processed(self, obs, acts, next_obs, dones):
        raw_rew = self.base_reward_net.predict_processed(obs, acts, next_obs, dones)
        return np.clip(raw_rew + self.bonus, a_min=self.floor, a_max=None)


def make_env_fn(env_id, config, svo_angle=None, seed=None):
    def _init():
        env = gym.make(env_id, config=config)
        if svo_angle is not None:
            svo_rad = np.deg2rad(svo_angle)
            env = SVOPureWrapper(env, svo_alpha=svo_rad)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_unimodal_airl(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="airl_no_bc",
        sync_tensorboard=True,
        config=vars(args),
        name=f"AIRL_{args.svo_angle}deg_simple",
    )

    config = ENV_CONFIG.copy()

    # Collect expert demonstrations
    print("Loading expert model and collecting demonstrations...")
    expert_model = DQN.load(args.model_path)
    expert_env = DummyVecEnv([
        make_env_fn(config["id"], config, svo_angle=args.svo_angle, seed=args.seed)
    ])

    expert_demos = rollout.rollout(
        expert_model,
        expert_env,
        rollout.make_sample_until(min_episodes=args.n_demos),
        rng=np.random.default_rng(args.seed),
        unwrap=False,
    )
    print(f"Collected {len(expert_demos)} expert demonstrations")
    expert_env.close()

    # Setup logging
    log_dir = f"{args.output_dir}/{run.id}"
    os.makedirs(log_dir, exist_ok=True)
    custom_logger = imitation_logger.configure(
        log_dir,
        ["stdout", "tensorboard", "wandb"]
    )

    # Create environments
    venv = SubprocVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + i)
        for i in range(args.n_envs)
    ])

    eval_env = DummyVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + 1000)
    ])

    # Build reward network
    base_reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        reward_hid_sizes=(64, 64),
    )

    reward_net = SurvivalBonusWrapper(
        base_reward_net,
        bonus=args.survival_bonus,
        floor=args.reward_floor
    )

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Initialize PPO - NO BC PRETRAINING to avoid device issues
    # But use better hyperparameters to compensate
    learner = PPO(
        env=venv,
        policy="MlpPolicy",
        batch_size=args.ppo_batch_size,
        ent_coef=args.entropy_coef,
        learning_rate=args.ppo_lr,
        n_steps=args.ppo_n_steps,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=args.seed,
        device=device,
    )
    print("Initialized PPO without BC pre-training")

    # Setup discriminator optimizer with regularization
    disc_opt = optim.Adam(
        reward_net.parameters(),
        lr=args.disc_lr,
        weight_decay=args.disc_weight_decay,
        betas=(0.5, 0.999)
    )

    # Initialize AIRL with conservative settings for no-BC training
    print("Initializing AIRL trainer...")
    airl_trainer = AIRL(
        demonstrations=expert_demos,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        demo_batch_size=args.demo_batch_size,
        gen_train_timesteps=args.gen_train_timesteps,
        n_disc_updates_per_round=args.n_disc_updates,
        custom_logger=custom_logger,
        allow_variable_horizon=True,
    )

    airl_trainer.disc_opt = disc_opt

    # Setup evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Train AIRL
    print("Starting AIRL training...")
    print(f"Config: {args.gen_train_timesteps} gen steps / {args.n_disc_updates} disc updates")
    print(f"Survival bonus: {args.survival_bonus}, Floor: {args.reward_floor}")

    airl_trainer.train(total_timesteps=args.total_timesteps)

    # Save models
    save_path = os.path.join(args.output_dir, f"airl_final_{args.svo_angle}")
    os.makedirs(save_path, exist_ok=True)
    learner.save(os.path.join(save_path, "gen_policy"))
    torch.save(reward_net.state_dict(), os.path.join(save_path, "reward_net.pth"))

    venv.close()
    eval_env.close()
    run.finish()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplified AIRL training (no BC) for Highway"
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--svo_angle", type=float, default=45.0)
    parser.add_argument("--n_demos", type=int, default=100)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # PPO hyperparameters - more conservative for no-BC training
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.02,
                        help="Slightly higher entropy for no-BC training")

    # AIRL hyperparameters - more generator-heavy without BC
    parser.add_argument("--gen_train_timesteps", type=int, default=16384,
                        help="More gen training without BC warm start")
    parser.add_argument("--n_disc_updates", type=int, default=1,
                        help="Fewer disc updates to avoid overwhelming random policy")
    parser.add_argument("--demo_batch_size", type=int, default=256)

    # Discriminator - slower and more regularized
    parser.add_argument("--disc_lr", type=float, default=1e-4,
                        help="Slower disc to not overwhelm random init")
    parser.add_argument("--disc_weight_decay", type=float, default=1e-3,
                        help="Heavy regularization")

    # Reward shaping - more aggressive without BC
    parser.add_argument("--survival_bonus", type=float, default=1.5,
                        help="Higher bonus to keep random policy alive")
    parser.add_argument("--reward_floor", type=float, default=-0.2)

    # Logging
    parser.add_argument("--output_dir", type=str, default="runs/airl_simple")
    parser.add_argument("--wandb_project", type=str, default="highway-airl")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    train_unimodal_airl(args)