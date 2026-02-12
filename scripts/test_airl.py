import os
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from collections import deque

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout, types
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


class AdaptiveAIRLMonitor(BaseCallback):
    """
    Monitors discriminator and adaptively adjusts training to prevent overfitting.
    This is THE KEY to preventing the 500k collapse.
    """

    def __init__(self, airl_trainer, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.airl_trainer = airl_trainer
        # CRITICAL FIX 1: Manually set the model so _on_step can access ep_info_buffer
        self.model = airl_trainer.gen_algo
        self.check_freq = check_freq

        # Track metrics
        self.disc_acc_gen_history = deque(maxlen=20)
        self.disc_acc_expert_history = deque(maxlen=20)
        self.disc_loss_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=50)

        # Store original settings
        self.n_disc_updates_original = airl_trainer.n_disc_updates_per_round
        self.disc_lr_original = airl_trainer.disc_opt.param_groups[0]['lr']
        self.weight_decay_original = airl_trainer.disc_opt.param_groups[0]['weight_decay']

        # State tracking
        self.gen_competitive_rounds = 0
        self.overfitting_detected = False
        self.last_intervention_step = 0

        # For metric tracking
        self.last_logged_metrics = {}

    # CRITICAL FIX 2: Make the object callable to satisfy AIRL.train(callback=...)
    def __call__(self, round_num):
        """
        Bridge between imitation's callback(round_num) and SB3's callback structure.
        """
        # Sync the timestep counter with the generator
        self.num_timesteps = self.model.num_timesteps

        # Increment call counter manually
        self.n_calls += 1

        # Execute the logic
        return self._on_step()

    def _on_step(self):
        # Collect episode rewards
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.reward_history.append(ep_info['r'])

        # Try to get discriminator metrics from wandb
        try:
            import wandb
            if wandb.run is not None:
                # Get latest metrics from wandb history
                history = wandb.run.history(samples=1, pandas=False)
                if history:
                    latest = list(history)[-1] if isinstance(history, list) else history

                    if 'disc/disc_acc_gen' in latest:
                        self.disc_acc_gen_history.append(latest['disc/disc_acc_gen'])
                        self.last_logged_metrics['gen_acc'] = latest['disc/disc_acc_gen']

                    if 'disc/disc_acc_expert' in latest:
                        self.disc_acc_expert_history.append(latest['disc/disc_acc_expert'])
                        self.last_logged_metrics['expert_acc'] = latest['disc/disc_acc_expert']

                    if 'disc/disc_loss' in latest:
                        self.disc_loss_history.append(latest['disc/disc_loss'])
                        self.last_logged_metrics['disc_loss'] = latest['disc/disc_loss']
        except Exception as e:
            if self.verbose > 1:
                print(f"Note: Could not fetch wandb metrics: {e}")

        # Check every N steps
        if self.n_calls % self.check_freq == 0 and self.n_calls > 0:
            self._adaptive_adjustment()

        return True

    def _adaptive_adjustment(self):
        """The magic happens here - adaptive discriminator management."""

        if len(self.disc_acc_gen_history) < 5:
            if self.verbose > 1:
                print(f"   [Monitor] Collecting metrics... ({len(self.disc_acc_gen_history)}/5)")
            return  # Not enough data yet

        recent_gen_acc = np.mean(list(self.disc_acc_gen_history)[-5:])
        recent_expert_acc = np.mean(list(self.disc_acc_expert_history)[-5:])
        recent_disc_loss = np.mean(list(self.disc_loss_history)[-5:]) if len(self.disc_loss_history) > 0 else 0.5

        if self.verbose > 1:
            print(
                f"\n   [Monitor @ {self.num_timesteps}] Gen acc: {recent_gen_acc:.1%}, Expert acc: {recent_expert_acc:.1%}, Loss: {recent_disc_loss:.3f}")

        # === CRITICAL DETECTION: Generator becoming competitive ===
        if recent_gen_acc > 0.15 and not self.overfitting_detected:
            self.gen_competitive_rounds += 1

            if self.gen_competitive_rounds >= 2:  # Sustained improvement
                if self.verbose:
                    print("\n" + "=" * 70)
                    print("🎉 SUCCESS SIGNAL: Generator is competitive with expert!")
                    print(f"   Gen accuracy: {recent_gen_acc:.1%} (above 15% threshold)")
                    print("   REDUCING discriminator training to prevent overfitting...")
                    print("=" * 70 + "\n")

                # INTERVENTION 1: Reduce discriminator updates
                self.airl_trainer.n_disc_updates_per_round = 1

                # INTERVENTION 2: Increase weight decay significantly
                for param_group in self.airl_trainer.disc_opt.param_groups:
                    param_group['weight_decay'] = min(
                        param_group['weight_decay'] * 3,
                        0.05
                    )

                # INTERVENTION 3: Reduce discriminator learning rate
                for param_group in self.airl_trainer.disc_opt.param_groups:
                    param_group['lr'] *= 0.5

                self.last_intervention_step = self.num_timesteps
                self.gen_competitive_rounds = 0

                if self.verbose:
                    print(f"   New settings:")
                    print(f"   - Disc updates: {self.airl_trainer.n_disc_updates_per_round}")
                    print(f"   - Weight decay: {param_group['weight_decay']:.4f}")
                    print(f"   - Learning rate: {param_group['lr']:.6f}\n")

        else:
            self.gen_competitive_rounds = 0

        # === DETECTION: Discriminator overfitting ===
        if recent_expert_acc > 0.98 and recent_gen_acc < 0.12:
            if self.verbose:
                print("\n" + "=" * 70)
                print("⚠️  WARNING: Discriminator overfitting detected!")
                print(f"   Expert acc: {recent_expert_acc:.1%} (too high)")
                print(f"   Gen acc: {recent_gen_acc:.1%} (too low)")
                print("   Taking emergency corrective action...")
                print("=" * 70 + "\n")

            self.overfitting_detected = True

            # EMERGENCY INTERVENTION
            if self.num_timesteps - self.last_intervention_step > 50000:
                # Temporarily freeze discriminator
                self.airl_trainer.n_disc_updates_per_round = 0

                if self.verbose:
                    print("   🚨 FREEZING discriminator for next 50k steps")
                    print("      to let generator catch up!\n")

                self.last_intervention_step = self.num_timesteps

        # === RECOVERY: Unfreeze if generator improves ===
        if self.airl_trainer.n_disc_updates_per_round == 0:
            if recent_gen_acc > 0.15 or self.num_timesteps - self.last_intervention_step > 100000:
                if self.verbose:
                    print("\n   ✅ Unfreezing discriminator (gen_acc improved or timeout)")
                self.airl_trainer.n_disc_updates_per_round = 1
                self.overfitting_detected = False

        # === DETECTION: Training collapse ===
        if len(self.reward_history) > 20:
            recent_rewards = list(self.reward_history)[-20:]
            reward_trend = np.mean(recent_rewards[-10:]) - np.mean(recent_rewards[:10])

            if reward_trend < -5 and self.num_timesteps > 300000:
                if self.verbose:
                    print("\n" + "=" * 70)
                    print("🚨 COLLAPSE DETECTED: Rewards are dropping significantly!")
                    print(f"   Recent reward trend: {reward_trend:.2f}")
                    print("   Applying aggressive interventions...")
                    print("=" * 70 + "\n")

                # NUCLEAR OPTION: Massive regularization
                for param_group in self.airl_trainer.disc_opt.param_groups:
                    param_group['weight_decay'] = 0.1

                self.airl_trainer.n_disc_updates_per_round = 0
                self.last_intervention_step = self.num_timesteps

        # Log current state
        try:
            wandb.log({
                "monitor/disc_acc_gen": recent_gen_acc,
                "monitor/disc_acc_expert": recent_expert_acc,
                "monitor/disc_loss": recent_disc_loss,
                "monitor/n_disc_updates": self.airl_trainer.n_disc_updates_per_round,
                "monitor/disc_weight_decay": self.airl_trainer.disc_opt.param_groups[0]['weight_decay'],
                "monitor/disc_lr": self.airl_trainer.disc_opt.param_groups[0]['lr'],
                "monitor/gen_competitive": self.gen_competitive_rounds,
                "monitor/overfitting_detected": int(self.overfitting_detected),
            }, step=self.num_timesteps)
        except:
            pass



def augment_demonstrations(demos, noise_scale=0.02, augment_fraction=0.5):
    """
    Augment expert demonstrations with noise to prevent memorization.
    This is CRITICAL for preventing the 500k overfitting.
    """
    augmented = list(demos)  # Keep originals
    n_augment = int(len(demos) * augment_fraction)

    for i in range(n_augment):
        traj = demos[i % len(demos)]

        # Add Gaussian noise to observations
        noise = np.random.normal(0, noise_scale, traj.obs.shape)
        noisy_obs = traj.obs + noise

        augmented.append(types.Trajectory(
            obs=noisy_obs,
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal
        ))

    print(f"Augmented {n_augment} demonstrations (total: {len(augmented)})")
    return augmented


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
        job_type="airl_final_fixed",
        sync_tensorboard=True,
        config=vars(args),
        name=f"AIRL_{args.svo_angle}deg_FINAL",
    )

    config = ENV_CONFIG.copy()

    # Collect expert demonstrations
    print("=" * 70)
    print("LOADING EXPERT AND COLLECTING DEMONSTRATIONS")
    print("=" * 70)
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
    print(f"✓ Collected {len(expert_demos)} expert demonstrations")
    expert_env.close()

    # CRITICAL: Augment demonstrations
    expert_demos = augment_demonstrations(
        expert_demos,
        noise_scale=args.demo_noise_scale,
        augment_fraction=args.demo_augment_fraction
    )

    # Setup logging
    log_dir = f"{args.output_dir}/{run.id}"
    os.makedirs(log_dir, exist_ok=True)
    custom_logger = imitation_logger.configure(
        log_dir,
        ["stdout", "tensorboard", "wandb"]
    )

    # Create environments
    print("\nCreating training environments...")
    venv = SubprocVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + i)
        for i in range(args.n_envs)
    ])
    print(f"✓ Created {args.n_envs} parallel environments")

    # Build reward network
    print("\nBuilding reward network...")
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
    print(f"✓ Reward network created (bonus={args.survival_bonus}, floor={args.reward_floor})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    # Initialize PPO
    print("\nInitializing PPO generator...")
    # learner = PPO(
    #     env=venv,
    #     policy="MlpPolicy",
    #     batch_size=args.ppo_batch_size,
    #     ent_coef=args.entropy_coef,
    #     learning_rate=args.ppo_lr,
    #     n_steps=args.ppo_n_steps,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     max_grad_norm=0.5,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     seed=args.seed,
    #     device=device,
    # )

    learner = SAC(

    )

    print("✓ PPO initialized")

    # Setup discriminator optimizer with HEAVY regularization
    print("\nConfiguring discriminator optimizer...")
    disc_opt = optim.Adam(
        reward_net.parameters(),
        lr=args.disc_lr,
        weight_decay=args.disc_weight_decay,
        betas=(0.5, 0.999)  # Lower momentum for stability
    )
    print(f"✓ Discriminator optimizer:")
    print(f"  - Learning rate: {args.disc_lr}")
    print(f"  - Weight decay: {args.disc_weight_decay}")
    print(f"  - Betas: (0.5, 0.999)")

    # Initialize AIRL
    print("\nInitializing AIRL trainer...")
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
    print("✓ AIRL trainer initialized")

    # THE KEY: Add adaptive monitoring callback
    print("\nSetting up adaptive discriminator monitor...")
    adaptive_monitor = AdaptiveAIRLMonitor(
        airl_trainer,
        check_freq=5000,
        verbose=1
    )


    # Print final configuration
    print("\n" + "=" * 70)
    print("FINAL CONFIGURATION - OPTIMIZED TO PREVENT 500K COLLAPSE")
    print("=" * 70)
    print(f"Total timesteps:          {args.total_timesteps:,}")
    print(f"Generator steps/round:    {args.gen_train_timesteps}")
    print(f"Discriminator updates:    {args.n_disc_updates}")
    print(f"Demo batch size:          {args.demo_batch_size}")
    print(f"Discriminator LR:         {args.disc_lr}")
    print(f"Discriminator weight decay: {args.disc_weight_decay}")
    print(f"Survival bonus:           {args.survival_bonus}")
    print(f"Demo augmentation:        ON ({args.demo_augment_fraction * 100:.0f}%)")
    print(f"Adaptive monitoring:      ON (checks every 5k steps)")
    print("=" * 70)
    print("\n🚀 Starting AIRL training...\n")

    # Train AIRL with callback
    airl_trainer.train(
        total_timesteps=args.total_timesteps,
        callback=adaptive_monitor
    )

    # Save models
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SAVING MODELS")
    print("=" * 70)
    save_path = os.path.join(args.output_dir, f"airl_final_{args.svo_angle}")
    os.makedirs(save_path, exist_ok=True)
    learner.save(os.path.join(save_path, "gen_policy"))
    torch.save(reward_net.state_dict(), os.path.join(save_path, "reward_net.pth"))
    print(f"✓ Models saved to: {save_path}")

    venv.close()
    run.finish()
    print("\n✅ Training pipeline complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FINAL AIRL Training Script - Optimized to Prevent 500k Collapse"
    )

    # Required
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to expert DQN model")

    # Environment
    parser.add_argument("--svo_angle", type=float, default=45.0)
    parser.add_argument("--n_demos", type=int, default=100)
    parser.add_argument("--total_timesteps", type=int, default=1_500_000,
                        help="Reduced from 2M since we're more efficient now")
    parser.add_argument("--n_envs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # PPO - Standard settings
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # AIRL - OPTIMIZED for 500k transition
    parser.add_argument("--gen_train_timesteps", type=int, default=16384,
                        help="More generator training per round")
    parser.add_argument("--n_disc_updates", type=int, default=2,
                        help="Start with 2, will be reduced adaptively")
    parser.add_argument("--demo_batch_size", type=int, default=256)

    # Discriminator - HEAVY REGULARIZATION
    parser.add_argument("--disc_lr", type=float, default=3e-5)
    parser.add_argument("--disc_weight_decay", type=float, default=0.001,
                        help="weight decay - key to preventing overfitting")

    # Reward shaping
    parser.add_argument("--survival_bonus", type=float, default=1.0)
    parser.add_argument("--reward_floor", type=float, default=-0.3)

    # Demo augmentation - CRITICAL
    parser.add_argument("--demo_noise_scale", type=float, default=0.025,
                        help="Noise scale for augmentation")
    parser.add_argument("--demo_augment_fraction", type=float, default=0.5,
                        help="Augment 50% of demos")

    # Logging
    parser.add_argument("--output_dir", type=str, default="runs/airl_final")
    parser.add_argument("--wandb_project", type=str, default="highway-airl")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    train_unimodal_airl(args)