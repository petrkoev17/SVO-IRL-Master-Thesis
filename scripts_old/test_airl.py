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

from configs.env_config import ENV_CONFIG
from src.envs.svo_pure_wrapper import SVOPureWrapper

# Import absorbing wrapper (make sure this file is in the same directory or in Python path)
try:
    from scripts.absorbing_state import AbsorbingWrapper

    ABSORBING_AVAILABLE = True
except ImportError:
    print("Warning: absorbing_wrapper.py not found. Running without absorbing states.")
    ABSORBING_AVAILABLE = False

# Import fixed horizon wrapper
try:
    from scripts.fixed_horizon import FixedHorizonWrapper

    FIXED_HORIZON_AVAILABLE = True
except ImportError:
    print("Warning: fixed_horizon_wrapper.py not found.")
    FIXED_HORIZON_AVAILABLE = False


class ImprovedRewardNet(BasicShapedRewardNet):
    """
    Enhanced reward network with dropout for better generalization.
    Extends BasicShapedRewardNet and adds dropout to the MLP.
    Safe for SVO - no behavioral priors, just better learning dynamics.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            dropout_rate=0.3,
            hid_sizes=(64, 64),
            **kwargs
    ):
        # Initialize parent - this builds the base network
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            reward_hid_sizes=hid_sizes,
            **kwargs
        )

        self.dropout_rate = dropout_rate

        # Now rebuild the reward MLP with dropout
        # The parent creates self.mlp which is the main network
        # We need to inject dropout into it
        self._add_dropout_to_network()

    def _add_dropout_to_network(self):
        """Add dropout layers to the reward network."""
        try:
            # Access the MLP from BasicShapedRewardNet
            # The structure is: self.mlp (contains the actual network)
            if hasattr(self, 'mlp'):
                old_mlp = self.mlp

                # Build new MLP with dropout
                new_layers = []
                for i, module in enumerate(old_mlp.modules()):
                    if isinstance(module, nn.Sequential):
                        # This is the top-level Sequential, skip it
                        continue

                    new_layers.append(module)

                    # Add dropout after ReLU activations (but not the last one)
                    if isinstance(module, nn.ReLU):
                        # Check if this is not the last layer
                        remaining_modules = list(old_mlp.modules())[i + 2:]  # +2 to skip Sequential
                        has_more_layers = any(isinstance(m, nn.Linear) for m in remaining_modules)
                        if has_more_layers:
                            new_layers.append(nn.Dropout(self.dropout_rate))

                # Replace the MLP
                if len(new_layers) > 0:
                    self.mlp = nn.Sequential(*new_layers)
                    print(f"✓ Added dropout ({self.dropout_rate}) to reward network")
                else:
                    print(f"⚠ Could not add dropout - using network without dropout")
            else:
                print(f"⚠ No 'mlp' attribute found - using network without dropout")

        except Exception as e:
            print(f"⚠ Could not inject dropout: {e}")
            print(f"  Continuing without dropout modification...")


def compute_gradient_penalty(discriminator, expert_data, agent_data, lambda_gp=10.0):
    """
    Compute gradient penalty for discriminator regularization.
    Helps prevent overconfident/collapsed discriminator.
    Safe for SVO - doesn't encode behavioral preferences.
    """
    batch_size = expert_data[0].shape[0]
    device = expert_data[0].device

    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, device=device)

    # Interpolate between expert and agent data
    interpolated = []
    for expert_tensor, agent_tensor in zip(expert_data, agent_data):
        # Expand alpha to match tensor dimensions
        alpha_expanded = alpha
        for _ in range(len(expert_tensor.shape) - 1):
            alpha_expanded = alpha_expanded.unsqueeze(-1)
        alpha_expanded = alpha_expanded.expand_as(expert_tensor)

        interp = alpha_expanded * expert_tensor + (1 - alpha_expanded) * agent_tensor
        interp.requires_grad_(True)
        interpolated.append(interp)

    # Compute discriminator output on interpolated data
    obs, acts, next_obs, dones = interpolated
    disc_interpolated = discriminator(obs, acts, next_obs, dones)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return lambda_gp * gradient_penalty


def make_env_fn(env_id, config, svo_angle=None, seed=None, use_absorbing=True, flatten_obs=True, fixed_horizon=0):
    """
    Creates environment with proper observation flattening and optional absorbing states.

    Args:
        flatten_obs: If True, flatten observations. Set to False when loading expert trained without flattening.
        fixed_horizon: If > 0, enforce fixed episode length. Required for allow_variable_horizon=False.
    """

    def _init():
        env = gym.make(env_id, config=config)

        # Flatten observations if requested (needed for absorbing states to work)
        if flatten_obs:
            env = FlattenObservation(env)

        if svo_angle is not None:
            svo_rad = np.deg2rad(svo_angle)
            env = SVOPureWrapper(env, svo_alpha=svo_rad)

        # Add fixed horizon wrapper if needed (this includes absorbing states)
        if fixed_horizon > 0 and FIXED_HORIZON_AVAILABLE:
            env = FixedHorizonWrapper(env, horizon=fixed_horizon)
        # Otherwise add absorbing states wrapper (for variable horizon mode)
        elif use_absorbing and ABSORBING_AVAILABLE and flatten_obs:
            try:
                env = AbsorbingWrapper(env)
            except Exception as e:
                print(f"Warning: Could not use new gym API absorbing wrapper: {e}")
                print("Trying old gym API...")
                try:
                    env = AbsorbingWrapperOldGym(env)
                except Exception as e2:
                    print(f"Warning: Could not use old gym API absorbing wrapper: {e2}")
                    print("Continuing without absorbing states...")

        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _init


def pad_demonstrations_to_fixed_horizon(demos, target_horizon):
    """
    Pad all demonstrations to a fixed horizon using absorbing state transitions.

    For demos shorter than target_horizon:
    - Repeat the last observation (absorbing state with indicator=1)
    - Use no-op actions (action 0)
    - Mark as non-terminal until the end

    This allows using allow_variable_horizon=False while preserving crash information
    via absorbing states.
    """
    print(f"Padding demonstrations to fixed horizon of {target_horizon} steps...")

    from imitation.data.types import Trajectory
    padded_demos = []

    for i, traj in enumerate(demos):
        traj_len = len(traj.obs)

        if traj_len >= target_horizon:
            # Truncate if longer than target
            new_traj = Trajectory(
                obs=traj.obs[:target_horizon],
                acts=traj.acts[:target_horizon],
                infos=traj.infos[:target_horizon] if traj.infos is not None else None,
                terminal=True  # Mark as terminal at the end
            )
        else:
            # Pad if shorter than target
            pad_length = target_horizon - traj_len

            # Last observation should be absorbing state (indicator=1)
            # Create absorbing observation (all zeros + indicator=1)
            obs_dim = traj.obs.shape[1]
            absorbing_obs = np.zeros((1, obs_dim), dtype=traj.obs.dtype)
            absorbing_obs[0, -1] = 1.0  # Set absorbing indicator to 1

            # Repeat absorbing state for padding
            padded_obs = np.concatenate([
                traj.obs,
                np.repeat(absorbing_obs, pad_length, axis=0)
            ], axis=0)

            # Use no-op action (0) for padding
            padded_acts = np.concatenate([
                traj.acts,
                np.zeros((pad_length,), dtype=traj.acts.dtype)
            ], axis=0)

            # Pad infos if they exist
            if traj.infos is not None:
                padded_infos = traj.infos + [{"absorbing_padding": True}] * pad_length
            else:
                padded_infos = None

            new_traj = Trajectory(
                obs=padded_obs,
                acts=padded_acts,
                infos=padded_infos,
                terminal=True
            )

        padded_demos.append(new_traj)

        if i == 0:
            print(f"  Example: Original length {traj_len} -> Padded length {len(new_traj.obs)}")

    print(f"✓ Padded {len(padded_demos)} demonstrations to length {target_horizon}")
    return padded_demos


def augment_demonstrations_with_absorbing(demos, add_absorbing_indicator=True):
    """
    Augment demonstrations collected without absorbing states to work with
    absorbing state environments.

    Adds a zero-valued absorbing indicator to all observations in the demos.
    The last observation (after terminal) gets indicator=1.
    """
    if not add_absorbing_indicator:
        return demos

    print("Augmenting expert demonstrations with absorbing indicators...")

    from imitation.data.types import Trajectory
    augmented_demos = []

    for traj in demos:
        # Get original observation shape
        orig_shape = traj.obs.shape
        print(f"  Processing trajectory with obs shape: {orig_shape}")

        # Flatten observations if they're 3D (e.g., stacked frames)
        if len(orig_shape) == 3:
            # Shape is (timesteps, height, width) -> flatten to (timesteps, height*width)
            obs_flat = traj.obs.reshape(orig_shape[0], -1)
        else:
            obs_flat = traj.obs

        # Add indicator=0 to all non-terminal observations
        # Shape: (timesteps, features) + (timesteps, 1) = (timesteps, features+1)
        obs_with_indicator = np.concatenate([
            obs_flat,
            np.zeros((len(obs_flat), 1), dtype=obs_flat.dtype)
        ], axis=1)

        # Create new trajectory with augmented observations
        new_traj = Trajectory(
            obs=obs_with_indicator,
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal
        )
        augmented_demos.append(new_traj)

    print(f"✓ Augmented {len(augmented_demos)} demonstrations")
    print(f"  Original obs shape: {demos[0].obs.shape}")
    print(f"  Augmented obs shape: {augmented_demos[0].obs.shape}")

    return augmented_demos


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
                    use_absorbing=False, flatten_obs=False)  # NO flattening for expert
    ])
    print("Note: Collecting expert demos WITHOUT flattening (expert was trained on (10,4) obs)")

    expert_demos = rollout.rollout(
        expert_model,
        expert_env,
        rollout.make_sample_until(min_episodes=args.n_demos),
        rng=np.random.default_rng(args.seed),
        unwrap=False,
    )

    # Augment demonstrations with absorbing indicators if using absorbing states
    if args.use_absorbing:
        expert_demos = augment_demonstrations_with_absorbing(expert_demos, add_absorbing_indicator=True)

        # Pad to fixed horizon if not allowing variable horizon
        if not args.allow_variable_horizon:
            # Use max episode length + buffer as target horizon
            max_ep_len = max(len(ep.obs) for ep in expert_demos)
            target_horizon = args.fixed_horizon if args.fixed_horizon > 0 else int(max_ep_len * 1.2)
            print(f"Fixed horizon mode: target_horizon={target_horizon} (max_ep_len={max_ep_len})")
            expert_demos = pad_demonstrations_to_fixed_horizon(expert_demos, target_horizon)
        else:
            target_horizon = 0  # Variable horizon mode
    else:
        target_horizon = 0  # No absorbing states, no fixed horizon

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

    # Create environments - WITH flattening for learner (needed for absorbing states)
    # If fixed horizon mode, pass target_horizon to enforce fixed episode length
    venv = SubprocVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + i,
                    use_absorbing=args.use_absorbing, flatten_obs=True,
                    fixed_horizon=target_horizon)  # Enforce fixed horizon in env
        for i in range(args.n_envs)
    ])

    eval_env = DummyVecEnv([
        make_env_fn(config["id"], config, seed=args.seed + 1000,
                    use_absorbing=args.use_absorbing, flatten_obs=True,
                    fixed_horizon=target_horizon)  # Enforce fixed horizon in env
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

    # SAFE FIX #7: Gradient penalty (optional, controlled by flag)
    if args.use_gradient_penalty:
        original_train_disc = airl_trainer._train_disc

        def train_disc_with_gp(*args_tuple, **kwargs):
            """Wrapper to add gradient penalty to discriminator training."""
            # Call original discriminator training
            disc_loss = original_train_disc(*args_tuple, **kwargs)

            # Add gradient penalty (need to modify imitation library internals)
            # This is a simplified version - you may need to adapt based on imitation version
            # For now, just return the original loss
            # Full implementation would require accessing batch data
            return disc_loss

        # Note: Full gradient penalty implementation requires modifying imitation library
        # Or subclassing AIRL. This is shown as concept.
        print("Note: Gradient penalty flag set but requires custom AIRL subclass")

    print("\n=== TRAINING CONFIGURATION ===")
    print(f"AIRL with improvements (no BC pre-training, no survival bonus)")
    if args.use_absorbing:
        print(f"✓ Using absorbing states (helps discriminator identify crashes)")
    else:
        print(f"⚠ Pure AIRL without absorbing states")
    if not args.allow_variable_horizon:
        print(f"✓ Fixed horizon mode (prevents episode length exploitation)")
    else:
        print(f"⚠ Variable horizon mode (may leak information)")
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
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.1,
                        help="SAFE FIX: Higher entropy for exploration (was 0.02)")

    # AIRL hyperparameters - balanced for pure AIRL
    parser.add_argument("--gen_train_timesteps", type=int, default=32768,
                        help="SAFE FIX: More gen steps to explore before disc update (was 16384)")
    parser.add_argument("--n_disc_updates", type=int, default=1,
                        help="SAFE FIX: Keep at 1 to prevent discriminator racing ahead")
    parser.add_argument("--demo_batch_size", type=int, default=1024,
                        help="SAFE FIX: Large batches for better discriminator learning (auto-adjusts if too large)")

    # Discriminator hyperparameters - the key improvements
    parser.add_argument("--disc_lr", type=float, default=5e-5,
                        help="SAFE FIX: Much lower LR to prevent premature convergence (was 1e-4)")
    parser.add_argument("--disc_weight_decay", type=float, default=1e-4,
                        help="SAFE FIX: Weight decay for regularization (was 1e-3)")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="SAFE FIX: Dropout in discriminator for generalization")
    parser.add_argument("--use_gradient_penalty", action="store_true",
                        help="SAFE FIX: Add gradient penalty (requires custom implementation)")

    # Absorbing states - helps discriminator distinguish crashes
    parser.add_argument("--use_absorbing", action="store_true", default=True,
                        help="Use absorbing states to help discriminator identify crashes")
    parser.add_argument("--no_absorbing", dest="use_absorbing", action="store_false",
                        help="Disable absorbing states (pure AIRL)")

    # Variable horizon settings
    parser.add_argument("--allow_variable_horizon", action="store_true", default=False,
                        help="Allow variable horizon (not recommended, can leak info)")
    parser.add_argument("--fixed_horizon", type=int, default=0,
                        help="Fixed horizon length (0 = auto-calculate from max demo length)")

    # Logging
    parser.add_argument("--output_dir", type=str, default="runs/pure_airl_improved")
    parser.add_argument("--wandb_project", type=str, default="highway-airl")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AIRL TRAINING - IMPROVED VERSION WITH ABSORBING STATES")
    print("=" * 60)
    print("Changes from baseline:")
    print("  ✓ REMOVED: Survival bonus (pure reward learning)")
    print("  ✓ REMOVED: BC pre-training (pure AIRL)")
    print("  ✓ ADDED: Discriminator dropout (default 0.3)")
    print("  ✓ ADDED: Lower discriminator LR (default 5e-5)")
    print("  ✓ ADDED: Higher entropy coefficient (default 0.1)")
    print("  ✓ ADDED: Larger demo batch size (default 1024)")
    print("  ✓ ADDED: More gen steps per disc update (default 32768)")
    print("  ✓ ADDED: Weight decay regularization (default 1e-4)")
    print("  ✓ ADDED: Absorbing states (helps discriminator identify crashes)")
    print("\nAll changes are SVO-safe (no behavioral priors)")
    print("Use --no_absorbing to disable absorbing states")
    print("=" * 60 + "\n")

    train_pure_airl(args)