"""
CartPole-v1 Sanity Check for IQ-Learn Port
============================================

Replicates the original Div-Infinity/IQ-Learn CartPole experiment:
    python train_iq.py agent=softq method=iq env=cartpole \
        expert.demos=1 expert.subsample_freq=20 \
        agent.init_temp=0.001 method.chi=True method.loss=value_expert

Uses the original repo's .npy expert data directly — no expert training needed.

Expected result: reaches expert-level reward of 500 within ~3-5k updates.

Usage:
    python cartpole_sanity_check.py --expert-npy path/to/iq_learn/experts/CartPole-v1_1000.npy
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
import pickle

# ---------------------------------------------------------------------------
# Add project root to path so we can import your IQLearnTrainer
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for candidate in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, '..'),
                  os.path.join(PROJECT_ROOT, '..', '..')]:
    src_path = os.path.join(candidate, 'src')
    if os.path.isdir(src_path):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

from src.algorithms.iq_learner import IQLearnTrainer


# ======================================================================
# Step 1: Load original expert data
# ======================================================================

def load_original_expert(expert_path, num_demos=1, subsample_freq=20, seed=42):
    """
    Load expert data from the original IQ-Learn repo's .npy or .pkl file
    and convert to the format your IQLearnTrainer expects:
        List[List[Tuple(state, action, reward, next_state, done, crashed, r_self, r_global)]]

    The original format is a dict with keys:
        states:      [num_trajs][traj_len, state_dim]
        next_states: [num_trajs][traj_len, state_dim]
        actions:     [num_trajs][traj_len]
        rewards:     [num_trajs][traj_len]
        dones:       [num_trajs][traj_len]
        lengths:     [num_trajs]
    """
    print("=" * 60)
    print("Step 1: Loading original expert data")
    print("=" * 60)

    # Load the file
    if expert_path.endswith('.pkl'):
        with open(expert_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = np.load(expert_path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()  # unwrap 0-d array wrapping a dict

    num_available = len(data['states'])
    print(f"  File: {expert_path}")
    print(f"  Available trajectories: {num_available}")

    # Sample `num_demos` trajectories (same as original ExpertDataset)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_available)
    selected_idx = perm[:num_demos]

    trajectories = []
    total_transitions = 0

    for i, idx in enumerate(selected_idx):
        states      = np.array(data['states'][idx])
        next_states = np.array(data['next_states'][idx])
        actions     = np.array(data['actions'][idx])
        rewards     = np.array(data['rewards'][idx])
        dones       = np.array(data['dones'][idx])

        traj_len = len(states)

        # Subsample every `subsample_freq` steps (same as original)
        trajectory = []
        for t in range(0, traj_len, subsample_freq):
            trajectory.append((
                states[t].astype(np.float32),
                int(actions[t]),
                float(rewards[t]),
                next_states[t].astype(np.float32),
                float(dones[t]),
                0.0,    # crashed
                0.0,    # r_self
                0.0,    # r_global
            ))

        trajectories.append(trajectory)
        total_transitions += len(trajectory)
        traj_reward = float(np.sum(rewards))
        print(f"  Demo {i} (traj #{idx}): {traj_len} steps, reward={traj_reward:.0f}, "
              f"kept {len(trajectory)} transitions (1/{subsample_freq})")

    print(f"  Total transitions in expert buffer: {total_transitions}\n")
    return trajectories


# ======================================================================
# Step 2: Train IQ-Learn
# ======================================================================

def train_iq_learn(expert_demos, num_updates=5000, eval_freq=200, seed=42):
    """
    Train your IQLearnTrainer on CartPole with hyperparameters matching the
    original repo's CartPole experiment.
    """
    print("=" * 60)
    print("Step 2: Training IQ-Learn (your implementation)")
    print("=" * 60)
    print("  Matching original settings:")
    print("    loss_type   = value_expert  (offline)")
    print("    divergence  = chi")
    print("    temperature = 0.001")
    print("    gamma       = 0.99")
    print("    No learner data (pure offline)")
    print()

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = IQLearnTrainer(
        env=env,
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dims=[64, 64],           # Matching original's OfflineQNetwork (64, 64)
        lr=1e-4,                        # Original default: critic_lr=1e-4
        gamma=0.99,
        tau=0.005,
        device='cpu',
        # IQ-Learn settings matching original CartPole experiment
        loss_type='value_expert',       # Offline: E_{expert}[V(s) - γV(s')]
        divergence='chi',               # χ² divergence
        div_alpha=0.5,                  # Default from original
        temperature=0.001,              # agent.init_temp=0.001
        use_target_network=True,
    )

    trainer.load_expert_demonstrations(expert_demos)

    # --- Training loop with periodic eval ---
    eval_updates = []
    eval_rewards = []
    eval_stds = []

    print(f"  Training for {num_updates} updates (eval every {eval_freq})...\n")
    t0 = time.time()

    for update in range(1, num_updates + 1):
        # Pure offline: only expert data, no learner data
        info = trainer.update(batch_size=32)     # Original default batch=32

        if update % eval_freq == 0:
            results = trainer.evaluate(num_episodes=10)
            eval_updates.append(update)
            eval_rewards.append(results['mean_reward'])
            eval_stds.append(results['std_reward'])

            elapsed = time.time() - t0
            print(f"  Update {update:5d}  |  Eval reward: {results['mean_reward']:7.1f} "
                  f"± {results['std_reward']:5.1f}  |  Loss: {info['loss']:.4f}  |  "
                  f"Time: {elapsed:.1f}s")

            # Early success check
            if results['mean_reward'] >= 490:
                print(f"\n  SUCCESS! Reached expert-level reward ({results['mean_reward']:.0f}) "
                      f"at update {update}.")
                break

    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed:.1f}s")

    env.close()
    return eval_updates, eval_rewards, eval_stds, trainer


# ======================================================================
# Step 3: Plot results
# ======================================================================

def plot_results(eval_updates, eval_rewards, eval_stds, save_path="cartpole_iq_learn_result.png"):
    print("\n" + "=" * 60)
    print("Step 3: Plotting results")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 5))

    rewards = np.array(eval_rewards)
    stds = np.array(eval_stds)

    ax.plot(eval_updates, rewards, 'b-', linewidth=2, label='IQ-Learn (your port)')
    ax.fill_between(eval_updates, rewards - stds, rewards + stds, alpha=0.2, color='b')
    ax.axhline(y=500, color='r', linestyle='--', linewidth=1.5, label='Expert level (500)')
    ax.axhline(y=490, color='orange', linestyle=':', linewidth=1, label='Success threshold (490)')

    ax.set_xlabel('Training Updates', fontsize=12)
    ax.set_ylabel('Eval Reward (10 episodes)', fontsize=12)
    ax.set_title('CartPole-v1 — IQ-Learn Sanity Check\n'
                 '(1 demo, subsample 20×, offline, χ² div, τ=0.001)',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 550])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to: {save_path}")
    plt.close()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CartPole IQ-Learn sanity check')
    parser.add_argument('--expert-npy', type=str, required=True,
                        help='Path to original IQ-Learn expert file '
                             '(e.g. iq_learn/experts/CartPole-v1_1000.npy or .pkl)')
    parser.add_argument('--num-demos', type=int, default=10,
                        help='Number of expert demos to use (default: 1)')
    parser.add_argument('--subsample-freq', type=int, default=20,
                        help='Subsample every N steps (default: 20)')
    parser.add_argument('--num-updates', type=int, default=5000,
                        help='Number of training updates (default: 5000)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. Load original expert data
    demos = load_original_expert(
        args.expert_npy,
        num_demos=args.num_demos,
        subsample_freq=args.subsample_freq,
        seed=args.seed + 42,  # Match original: seed=args.seed + 42
    )

    # 2. Train IQ-Learn
    updates, rewards, stds, trainer = train_iq_learn(
        expert_demos=demos,
        num_updates=args.num_updates,
        eval_freq=200,
        seed=args.seed,
    )

    # 3. Plot
    plot_results(updates, rewards, stds,
                 save_path="cartpole_iq_learn_result.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_reward = max(rewards) if rewards else 0
    best_update = updates[rewards.index(best_reward)] if rewards else 0
    print(f"  Best eval reward : {best_reward:.1f}  (at update {best_update})")
    print(f"  Target           : 500")
    if best_reward >= 490:
        print(f"  Result           : PASS — your IQ-Learn port works!")
    else:
        print(f"  Result           : DID NOT REACH 490. See notes below.")
        print(f"  Possible fixes   :")
        print(f"    - Try more updates (--num-updates 10000)")
        print(f"    - Try more demos (--num-demos 10)")
        print(f"    - Adjust temperature in the script")