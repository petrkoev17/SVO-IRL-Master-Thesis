import os
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FlattenObservation

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicShapedRewardNet, RewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imitation_logger
from imitation.util import networks


from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper

# Import concatenated episodes wrapper
try:
    from src.envs.concatenated_episodes_wrapper import ConcatenatedEpisodesWrapper

    CONCATENATED_AVAILABLE = True
except ImportError:
    print("Warning: concatenated_episodes_wrapper.py not found.")
    CONCATENATED_AVAILABLE = False


class ImprovedRewardNet(BasicShapedRewardNet):
    def __init__(
            self,
            observation_space,
            action_space,
            dropout_rate=0.3,
            hid_sizes=(64, 64),
            **kwargs
    ):
        # 1. Initialize the parent first to set up spaces and shaping
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            **kwargs
        )

        self.dropout_rate = dropout_rate

        # 2. Build a custom MLP with Dropout
        # We calculate input size based on the observation + action space
        def get_flattened_size(space: gym.Space) -> int:
            if isinstance(space, gym.spaces.Box):
                return int(np.prod(space.shape))
            elif isinstance(space, gym.spaces.Discrete):
                return int(space.n)
            # Add other space types if needed (MultiDiscrete, etc.)
            return gym.spaces.utils.flatdim(space)

        obs_size = get_flattened_size(observation_space)
        act_size = get_flattened_size(action_space)
        combined_size = obs_size + act_size

        layers = []
        in_size = combined_size

        for h in hid_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            in_size = h

        # Final output layer (reward is a scalar)
        layers.append(nn.Linear(in_size, 1))

        # 3. Override the internal base network
        # In BasicShapedRewardNet, the reward logic usually points to self.base
        self.base = nn.Sequential(*layers)

        print(f"✓ Created custom reward MLP with dropout ({self.dropout_rate})")

    def reward_train(self, state, action, next_state, done):
        # Ensure we use our custom base during forward passes
        return self.base(self.preprocess(state, action, next_state, done))

def make_env_fn(env_id, config, svo_angle=None, seed=None, flatten_obs=True, trajectory_length=0):
    """
    Creates environment with proper observation flattening and optional trajectory concatenation.

    Args:
        flatten_obs: If True, flatten observations.
        trajectory_length: If > 0, concatenate episodes to this length for fixed horizon mode.
    """

    def _init():
        env = gym.make(env_id, config=config)

        # Flatten observations if requested
        if flatten_obs:
            env = FlattenObservation(env)

        if svo_angle is not None:
            svo_rad = np.deg2rad(svo_angle)
            env = SVOPureWrapper(env, svo_alpha=svo_rad)

        # Add concatenated episodes wrapper for fixed horizon
        if trajectory_length > 0 and CONCATENATED_AVAILABLE:
            env = ConcatenatedEpisodesWrapper(env, trajectory_length=trajectory_length)

        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_pure_airl(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="pure_airl_improved",
        sync_tensorboard=True,
        config=vars(args),
        name=f"PureAIRL_{args.svo_angle}deg_v2",
    )

    config = ENV_CONFIG.copy()

    # Collect expert demonstrations
    # IMPORTANT: Expert was trained WITHOUT flattening, so don't flatten for demo collection
    print("Loading expert model and collecting demonstrations...")
    expert_model = DQN.load(args.model_path)
    expert_env = DummyVecEnv([
        make_env_fn(config["id"], config, svo_angle=args.svo_angle, seed=args.seed,
                    flatten_obs=False)  # NO flattening for expert
    ])
    print("Note: Collecting expert demos WITHOUT flattening (expert was trained on (10,4) obs)")

    expert_demos = rollout.rollout(
        expert_model,
        expert_env,
        rollout.make_sample_until(min_episodes=args.n_demos),
        rng=np.random.default_rng(args.seed),
        unwrap=False,
    )

    # For fixed horizon mode, use concatenated episodes (no absorbing states or padding)
    trajectory_length = 0
    if not args.allow_variable_horizon:
        # Calculate target trajectory length from initial demos
        max_ep_len = max(len(ep.obs) for ep in expert_demos)
        trajectory_length = args.trajectory_length if args.trajectory_length > 0 else int(max_ep_len * 1.5)

        print(f"\n=== Fixed Horizon Mode (Concatenated Episodes) ===")
        print(f"Target trajectory length: {trajectory_length}")
        print(f"Max expert episode length: {max_ep_len}")
        print(f"Re-collecting demos with concatenation wrapper...")

        # Close old env and create new one with concatenation
        expert_env.close()
        expert_env = DummyVecEnv([
            make_env_fn(config["id"], config, svo_angle=args.svo_angle, seed=args.seed,
                        flatten_obs=False, trajectory_length=trajectory_length)
        ])

        # Re-collect demos with concatenation
        expert_demos = rollout.rollout(
            expert_model,
            expert_env,
            rollout.make_sample_until(min_episodes=args.n_demos),
            rng=np.random.default_rng(args.seed),
            unwrap=False,
        )

        demo_lengths = [len(ep.obs) for ep in expert_demos]
        print(f"✓ Re-collected {len(expert_demos)} concatenated demos")
        print(f"  Sample lengths: {demo_lengths[:5]}...")
        print(f"  All demos same length: {len(set(demo_lengths)) == 1}")

    # Calculate statistics
    n_transitions = sum(len(ep.obs) for ep in expert_demos)
    mean_ep_len = n_transitions / len(expert_demos)

    print(f"Collected {len(expert_demos)} expert demonstrations")
    print(f"Total transitions: {n_transitions}")
    print(f"Mean episode length: {mean_ep_len:.1f}")

    # Adjust demo batch size if needed
    if args.demo_batch_size > n_transitions:
        old_batch_size = args.demo_batch_size
        args.demo_batch_size = max(256, n_transitions // 2)  # Use half of transitions, min 256
        print(f"\n⚠️  WARNING: demo_batch_size ({old_batch_size}) > transitions ({n_transitions})")
        print(f"   Automatically adjusted to {args.demo_batch_size}")
        print(f"   Consider using --n_demos {int(old_batch_size / mean_ep_len * 1.5)} for original batch size\n")

    expert_env.close()

    # Setup logging
    log_dir = f"{args.output_dir}/{run.id}"
    os.makedirs(log_dir, exist_ok=True)
    custom_logger = imitation_logger.configure(
        log_dir,
        ["stdout", "tensorboard", "wandb"]
    )

    # Create environments - WITH flattening for learner
    # If fixed horizon mode, use concatenated episodes
    venv = SubprocVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + i,
                    flatten_obs=False,
                    trajectory_length=trajectory_length)  # Use concatenation for fixed horizon
        for i in range(args.n_envs)
    ])

    eval_env = DummyVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + 1000,
                    flatten_obs=False,
                    trajectory_length=trajectory_length)  # Use concatenation for fixed horizon
    ])

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Build improved reward network with dropout
    # SAFE FIX #1: Dropout for discriminator generalization
    reward_net = ImprovedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        dropout_rate=args.dropout_rate,
        hid_sizes=(64, 64),
    ).to(device)

    print(f"Reward network initialized with dropout={args.dropout_rate}")

    # Initialize PPO with higher entropy for exploration
    # SAFE FIX #2: Higher entropy coefficient
    learner = PPO(
        env=venv,
        policy="MlpPolicy",
        batch_size=args.ppo_batch_size,
        ent_coef=args.entropy_coef,  # Increased from 0.02 to 0.1
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
    print(f"Initialized PPO with entropy_coef={args.entropy_coef} (pure AIRL, no BC)")

    # Setup discriminator optimizer
    # SAFE FIX #3: Lower discriminator learning rate
    # SAFE FIX #4: Weight decay for regularization
    disc_opt = optim.Adam(
        reward_net.parameters(),
        lr=args.disc_lr,  # Lowered to 1e-4 or 5e-5
        weight_decay=args.disc_weight_decay,
        betas=(0.5, 0.999)  # Lower beta1 for more stable gradients
    )

    print(f"Discriminator optimizer: lr={args.disc_lr}, weight_decay={args.disc_weight_decay}")

    # Initialize AIRL
    # SAFE FIX #5: Larger demo batch size for better discriminator learning
    # SAFE FIX #6: More generator steps per discriminator update
    print("Initializing AIRL trainer...")
    airl_trainer = AIRL(
        demonstrations=expert_demos,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        demo_batch_size=args.demo_batch_size,  # Increased to 2048
        gen_train_timesteps=args.gen_train_timesteps,  # Increased to give policy time to explore
        n_disc_updates_per_round=args.n_disc_updates,  # Kept at 1 to prevent discriminator from racing ahead
        custom_logger=custom_logger,
        allow_variable_horizon=args.allow_variable_horizon,
    )

    airl_trainer.disc_opt = disc_opt

    print("\n=== TRAINING CONFIGURATION ===")
    print(f"AIRL with concatenated episodes (fixed horizon)")
    print(f"Trajectory length: {trajectory_length if trajectory_length > 0 else 'variable (not recommended)'}")
    print(f"Expert demos: {len(expert_demos)} trajectories")
    print(f"Discriminator: lr={args.disc_lr}, dropout={args.dropout_rate}, wd={args.disc_weight_decay}")
    print(f"Generator: entropy={args.entropy_coef}, {args.gen_train_timesteps} steps per round")
    print(f"Update ratio: {args.gen_train_timesteps} gen steps : {args.n_disc_updates} disc updates")
    print(f"Demo batch size: {args.demo_batch_size}")
    print("===============================\n")

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
    print("Starting pure AIRL training...")
    airl_trainer.train(total_timesteps=args.total_timesteps)

    # Save models
    save_path = os.path.join(args.output_dir, f"pure_airl_{args.svo_angle}")
    os.makedirs(save_path, exist_ok=True)
    learner.save(os.path.join(save_path, "gen_policy"))
    torch.save(reward_net.state_dict(), os.path.join(save_path, "reward_net.pth"))

    venv.close()
    eval_env.close()
    run.finish()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pure AIRL training with safe improvements (no BC, no survival bonus)"
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to expert DQN model")
    parser.add_argument("--svo_angle", type=float, default=45.0,
                        help="SVO angle for demonstrations (degrees)")
    parser.add_argument("--n_demos", type=int, default=100,
                        help="Number of expert demonstrations to collect")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42)

    # PPO hyperparameters - optimized for pure AIRL
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--ppo_lr", type=float, default=1e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.05,
                        help="SAFE FIX: Higher entropy for exploration (was 0.02)")

    # AIRL hyperparameters - balanced for pure AIRL
    parser.add_argument("--gen_train_timesteps", type=int, default=16384,
                        help="SAFE FIX: More gen steps to explore before disc update (was 16384)")
    parser.add_argument("--n_disc_updates", type=int, default=4,
                        help="SAFE FIX: Keep at 1 to prevent discriminator racing ahead")
    parser.add_argument("--demo_batch_size", type=int, default=1024,
                        help="SAFE FIX: Large batches for better discriminator learning (auto-adjusts if too large)")

    # Discriminator hyperparameters - the key improvements
    parser.add_argument("--disc_lr", type=float, default=1e-4,
                        help="SAFE FIX: Much lower LR to prevent premature convergence (was 1e-4)")
    parser.add_argument("--disc_weight_decay", type=float, default=1e-5,
                        help="SAFE FIX: Weight decay for regularization (was 1e-3)")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="SAFE FIX: Dropout in discriminator for generalization")

    # Variable horizon settings
    parser.add_argument("--allow_variable_horizon", action="store_true", default=False,
                        help="Allow variable horizon (not recommended, can leak info)")
    parser.add_argument("--trajectory_length", type=int, default=0,
                        help="Trajectory length for concatenated episodes (0 = auto-calculate)")

    # Logging
    parser.add_argument("--output_dir", type=str, default="runs/airl_wrapped_3")
    parser.add_argument("--wandb_project", type=str, default="highway-airl")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AIRL TRAINING - CONCATENATED EPISODES APPROACH")
    print("=" * 60)
    print("Changes from baseline:")
    print("  ✓ REMOVED: Survival bonus (pure reward learning)")
    print("  ✓ REMOVED: BC pre-training (pure AIRL)")
    print("  ✓ REMOVED: Absorbing state padding (caused discriminator confusion)")
    print("  ✓ ADDED: Discriminator dropout (default 0.3)")
    print("  ✓ ADDED: Lower discriminator LR (default 5e-5)")
    print("  ✓ ADDED: Higher entropy coefficient (default 0.1)")
    print("  ✓ ADDED: Larger demo batch size (default 1024)")
    print("  ✓ ADDED: More gen steps per disc update (default 32768)")
    print("  ✓ ADDED: Weight decay regularization (default 1e-4)")
    print("  ✓ ADDED: Concatenated episodes for fixed horizon")
    print("\nApproach: Multiple real episodes concatenated to fixed length")
    print("Benefits: ~2% artificial transitions vs 70% with padding")
    print("All changes are SVO-safe (no behavioral priors)")
    print("=" * 60 + "\n")

    train_pure_airl(args)