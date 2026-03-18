import os
import math
import argparse
import pickle
import numpy as np
import torch
from datetime import datetime

# Import your custom environment and trainer
from src.envs.gridworld_v2.gridworld_v2 import SVOIntersectionGridV2 as SVOIntersectionEnv
from src.algorithms.iq_learner import IQLearnTrainer  # Ensure this points to the file containing your IQLearnTrainer class

KNOWN_AGENTS = {
    'egoistic': 0.0,
    'prosocial': math.pi / 4,
    'altruistic': math.pi / 2,
}


def parse_svo_alpha(value: str) -> float:
    if value in KNOWN_AGENTS:
        return KNOWN_AGENTS[value]
    return float(value)


def main():
    parser = argparse.ArgumentParser(description='Train IQ-Learn on SVO Gridworld')

    # Data
    parser.add_argument('--demo-path', type=str, required=True,
                        help='Path to the expert .pkl file')

    # SVO Regularization
    parser.add_argument('--svo-regularize', action='store_true',
                        help='Enable SVO regularization')
    parser.add_argument('--svo-mode', type=str, default='reward_reg',
                        choices=['bellman', 'reward_reg', 'reweight', 'reward_reg_reweight'])
    parser.add_argument('--svo-alpha', type=parse_svo_alpha, default='prosocial',
                        help='Target SVO profile (egoistic, prosocial, altruistic) or raw radians')
    parser.add_argument('--svo-lambda', type=float, default=1.0,
                        help='Strength of the SVO regularizer')
    parser.add_argument('--svo-reweight-temp', type=float, default=1.0,
                        help='Temperature for reweight mode softmax.')
    parser.add_argument('--normalize-svo', action='store_true',
                        help='Batch-normalize R_SVO to zero mean / unit var.')

    # Training params
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--output-dir', type=str, default='./runs',
                        help='Directory to save the trained model')
    parser.add_argument('--batch-size', type=int, default=64)  # Smaller batch for gridworld
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)


    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print(f"Initializing Gridworld IQ-Learn")
    print(f"{'=' * 50}")

    # 1. Initialize Environment
    env = SVOIntersectionEnv()
    state_dim = int(np.prod(env.observation_space.shape))
    print(state_dim)
    action_dim = env.action_space.n

    # 2. Load Expert Demonstrations
    print(f"Loading demonstrations from {args.demo_path}...")
    with open(args.demo_path, 'rb') as f:
        expert_trajectories = pickle.load(f)
    print(f"Loaded {len(expert_trajectories)} trajectories.")

    # 3. Initialize Trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = IQLearnTrainer(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        lr=args.lr,
        device=device,
        use_svo=args.svo_regularize,
        svo_mode=args.svo_mode,
        svo_alpha=args.svo_alpha,
        svo_lambda=args.svo_lambda,
        normalize_svo=args.normalize_svo,
    )

    trainer.load_expert_demonstrations(expert_trajectories)

    # 4. Training Loop
    print("\nStarting Training...")
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure the folder exists

    for update in range(1, args.updates + 1):
        info = trainer.update(batch_size=args.batch_size)

        if update % args.eval_freq == 0:
            eval_results = trainer.evaluate(num_episodes=10)
            print(f"[Update {update}/{args.updates}] "
                  f"Loss: {info['loss']:.4f} | "
                  f"Reward: {eval_results['mean_reward']:.2f} | "
                  f"Length: {eval_results['mean_length']:.1f}")

            if args.svo_regularize and 'svo_reg_loss' in info:
                print(f"    -> SVO Reg Loss: {info['svo_reg_loss']:.4f} | "
                      f"Recovered r̂: {info['recovered_reward_mean']:.3f}")

            # Save periodic checkpoints
            ckpt_path = os.path.join(args.output_dir, f"model_update_{update}.pt")
            trainer.save(ckpt_path)

    # Save the final model when training finishes
    final_path = os.path.join(args.output_dir, "prosocial.pt")
    trainer.save(final_path)
    print(f"\nTraining Complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()