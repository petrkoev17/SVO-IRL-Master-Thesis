"""
Main training script for IQ-Learn with highway-env.

Supports both:
  - Standard IQ-Learn baseline (default)
  - SVO-Regularized IQ-Learn (--svo-regularize)
"""

import gymnasium as gym
import highway_env
import numpy as np
import torch
import os
import argparse
from datetime import datetime
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.algorithms.iq_learner import IQLearnTrainer
from scripts.extract_demonstrations import (
    extract_from_multiple_agents,
    load_demonstrations,
    combine_demonstrations
)

from src.envs.svo_pure_wrapper import SVOPureWrapper
from configs.env_config import ENV_CONFIG

# from configs.intersection_config import INTERSECTION_CONFIG as ENV_CONFIG
# from src.envs.svo_intersection_wrapper import SVOIntersectionWrapper as SVOPureWrapper

def create_env(config: Dict, svo_angle: float = 0.0, render_mode: str = None):
    """
    Create highway environment with SVO wrapper.
    Args:
        svo_angle: SVO angle in radians. Wrapper is kept for statistics.
    """
    env = gym.make(config['id'], config=config, render_mode=render_mode)
    env = SVOPureWrapper(env, svo_alpha=svo_angle, lamb=1.0)
    return env


def plot_training_curves(trainer: IQLearnTrainer, save_dir: str):
    """Plot and save training curves locally"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    if trainer.losses:
        axes[0].plot(trainer.losses)
        axes[0].set_xlabel('Update Step')
        axes[0].set_ylabel('IQ-Learn Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)

    if trainer.expert_q_values:
        axes[1].plot(trainer.expert_q_values, label='Expert Q-values')
        if trainer.learner_q_values:
            axes[1].plot(trainer.learner_q_values, label='Learner Q-values')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Mean Q-value')
        axes[1].set_title('Q-value Evolution')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"Training curves saved to {save_dir}/training_curves.png")


def train_iq_learn(
    env_config: Dict,
    expert_demo_path: str,
    output_dir: str,
    # IQ-Learn hyperparameters
    num_updates: int = 10000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    temperature: float = 1.0,
    method: str = 'value',
    loss_type: str = 'v0',
    regularize_weight: float = 1.0,
    # SVO regularization
    use_svo: bool = False,
    svo_alpha: float = 0.0,
    svo_lambda: float = 1.0,
    normalize_svo: bool = False,
    # Training settings
    collect_learner_data: bool = False,
    learner_collection_freq: int = 1000,
    learner_rollout_steps: int = 1000,
    eval_freq: int = 500,
    eval_episodes: int = 10,
    save_freq: int = 2000,
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    # W&B
    use_wandb: bool = False,
    wandb_project: str = 'svo-irl',
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
):
    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Config dict
    # ------------------------------------------------------------------
    config_save = {
        'expert_demo_path': expert_demo_path,
        'num_updates': num_updates,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'tau': tau,
        'temperature': temperature,
        'method': method,
        'loss_type': loss_type,
        'regularize_weight': regularize_weight,
        'collect_learner_data': collect_learner_data,
        'seed': seed,
        # SVO
        'use_svo': use_svo,
        'svo_alpha': svo_alpha,
        'svo_alpha_deg': float(np.degrees(svo_alpha)) if use_svo else None,
        'svo_lambda': svo_lambda if use_svo else None,
        'normalize_svo': normalize_svo if use_svo else None,
        # Flatten env config
        **{f'env/{k}': v for k, v in env_config.items()
           if not isinstance(v, dict)},
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=4)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if use_wandb:
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config_save,
            tags=wandb_tags or [],
            dir=output_dir,
        )
        print(f"W&B run: {wandb.run.url}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    header = "SVO-Regularized IQ-Learn" if use_svo else "IQ-Learn (Baseline)"
    print(f"\n{'='*60}")
    print(f"{header} Training")
    print(f"{'='*60}")
    print(f"Output directory : {output_dir}")
    print(f"Device           : {device}")
    print(f"Expert demos     : {expert_demo_path}")
    if use_svo:
        print(f"SVO α_target     : {np.degrees(svo_alpha):.1f}° ({svo_alpha:.4f} rad)")
        print(f"SVO λ            : {svo_lambda}")
        print(f"SVO normalize    : {normalize_svo}")

    env = create_env(env_config, svo_angle=0.0)

    state_dim  = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    print(f"Obs shape        : {env.observation_space.shape} -> state_dim: {state_dim}")
    print(f"Action dim       : {action_dim}")

    # ------------------------------------------------------------------
    # Demonstrations
    # ------------------------------------------------------------------
    print(f"\nLoading expert demonstrations...")
    expert_trajectories, expert_stats, expert_metadata = load_demonstrations(expert_demo_path)
    print(f"Loaded {len(expert_trajectories)} trajectories")
    print(f"Expert stats: {expert_stats}")

    # Check if demonstrations contain SVO components
    if expert_trajectories and expert_trajectories[0]:
        sample_len = len(expert_trajectories[0][0])
        has_svo = sample_len >= 8  # (s, a, r, s', done, crashed, r_self, r_global)
        print(f"Transition format: {sample_len} elements ({'with' if has_svo else 'without'} SVO components)")
        if use_svo and not has_svo:
            print(
                "\n⚠  WARNING: SVO regularization is enabled but demonstrations "
                "lack r_self/r_global fields. Re-extract demonstrations using "
                "the updated extraction script. SVO shaping will have no effect."
            )

    if use_wandb:
        import wandb
        wandb.run.summary['expert/mean_reward']    = expert_stats['mean_reward']
        wandb.run.summary['expert/collision_rate'] = expert_stats['collision_rate']
        wandb.run.summary['expert/mean_length']    = expert_stats['mean_length']
        wandb.run.summary['expert/num_episodes']   = expert_stats['num_episodes']

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    print(f"\nInitializing trainer...")
    trainer = IQLearnTrainer(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        lr=learning_rate,
        gamma=gamma,
        tau=tau,
        device=device,
        method=method,
        loss_type=loss_type,
        regularize_weight=regularize_weight,
        temperature=temperature,
        # SVO
        use_svo=use_svo,
        svo_alpha=svo_alpha,
        svo_lambda=svo_lambda,
        normalize_svo=normalize_svo,
    )

    trainer.load_expert_demonstrations(expert_trajectories)

    if collect_learner_data:
        print("Prefilling learner with random exploration...")
        trainer.collect_learner_rollout(
            num_steps=2000,
            epsilon=1.0
        )

    # ------------------------------------------------------------------
    # Local training log
    # ------------------------------------------------------------------
    training_log = {
        'updates': [],
        'losses': [],
        'expert_q_means': [],
        'eval_rewards': [],
        'eval_collision_rates': [],
        'svo_reward_means': [],
        'abs_expert_q_means': [],
        'abs_gamma_v_means': [],
        'abs_svo_shift_means': [],
        'svo_shift_to_q_ratios': [],
    }

    # ------------------------------------------------------------------
    # Initial evaluation
    # ------------------------------------------------------------------
    print(f"\nInitial evaluation (untrained policy)...")
    eval_results = trainer.evaluate(num_episodes=eval_episodes)
    print(f"Initial performance: {eval_results}")

    training_log['updates'].append(0)
    training_log['eval_rewards'].append(eval_results['mean_reward'])
    training_log['eval_collision_rates'].append(eval_results['collision_rate'])

    if use_wandb:
        import wandb
        wandb.log({
            'eval/mean_reward':    eval_results['mean_reward'],
            'eval/std_reward':     eval_results['std_reward'],
            'eval/mean_length':    eval_results['mean_length'],
            'eval/collision_rate': eval_results['collision_rate'],
        }, step=0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting training for {num_updates} updates...")
    print(f"{'='*60}\n")

    pbar = tqdm(range(1, num_updates + 1), desc="Training")

    for update in pbar:

        # Optional: collect on-policy data
        if collect_learner_data and update % learner_collection_freq == 0:
            epsilon = max(0.1, 1.0 - update / num_updates)
            trainer.collect_learner_rollout(
                num_steps=learner_rollout_steps,
                epsilon=epsilon,
            )

        # Gradient update
        update_info = trainer.update(batch_size=batch_size)

        # Local log
        training_log['losses'].append(update_info['loss'])
        training_log['expert_q_means'].append(update_info['expert_q_mean'])
        training_log['abs_expert_q_means'].append(update_info.get('abs_expert_q_mean', 0.0))
        training_log['abs_gamma_v_means'].append(update_info.get('abs_gamma_v_mean', 0.0))

        if 'svo_raw_mean' in update_info:
            training_log['svo_reward_means'].append(update_info['svo_raw_mean'])
            training_log['abs_svo_shift_means'].append(update_info.get('abs_svo_shift_mean', 0.0))
            training_log['svo_shift_to_q_ratios'].append(update_info.get('svo_shift_to_q_ratio', 0.0))

        # W&B
        if use_wandb:
            import wandb
            log_dict = {
                'train/loss':           update_info['loss'],
                'train/expert_q_mean':  update_info['expert_q_mean'],
                'train/expert_v_mean':  update_info['expert_next_v_mean'],
                'train/abs_expert_q_mean': update_info.get('abs_expert_q_mean', 0.0),
                'train/abs_gamma_v_mean':  update_info.get('abs_gamma_v_mean', 0.0),
            }
            if 'learner_q_mean' in update_info:
                log_dict['train/learner_q_mean'] = update_info['learner_q_mean']
                log_dict['train/learner_v_mean'] = update_info['learner_next_v_mean']
            if 'svo_raw_mean' in update_info:
                log_dict['train/svo_raw_mean'] = update_info['svo_raw_mean']
                log_dict['train/svo_raw_std'] = update_info['svo_raw_std']
                log_dict['train/svo_shift_std'] = update_info['svo_shift_std']
                log_dict['train/svo_shift_min'] = update_info['svo_shift_min']
                log_dict['train/svo_shift_max'] = update_info['svo_shift_max']
                log_dict['train/abs_svo_shift_mean'] = update_info['abs_svo_shift_mean']
                log_dict['train/svo_shift_to_q_ratio'] = update_info['svo_shift_to_q_ratio']
            wandb.log(log_dict, step=update)

        # Progress bar
        if update % 10 == 0:
            desc = f"Loss: {update_info['loss']:.4f}  Expert Q: {update_info['expert_q_mean']:.3f}"
            if 'svo_raw_mean' in update_info:
                desc += f"  SVO raw: {update_info['svo_raw_mean']:.3f} shift:[{update_info['svo_shift_min']:.2f},{update_info['svo_shift_max']:.2f}]"
                desc += f"  SVO Ratio: {update_info['svo_shift_to_q_ratio']:.3f}"
            pbar.set_description(desc)

        # Periodic evaluation
        if update % eval_freq == 0:
            eval_results = trainer.evaluate(num_episodes=eval_episodes)

            training_log['updates'].append(update)
            training_log['eval_rewards'].append(eval_results['mean_reward'])
            training_log['eval_collision_rates'].append(eval_results['collision_rate'])

            print(f"\n[Update {update}/{num_updates}]")
            print(f"  Reward        : {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
            print(f"  Length        : {eval_results['mean_length']:.1f}")
            print(f"  Collision rate: {eval_results['collision_rate']:.2%}")
            print(f"  Loss          : {update_info['loss']:.4f}")
            print(f"  Expert Q      : {update_info['expert_q_mean']:.3f}")

            if 'svo_raw_mean' in update_info:
                print(f"  SVO raw       : {update_info['svo_raw_mean']:.3f} ± {update_info['svo_raw_std']:.3f}")
                print(f"  SVO shift     : [{update_info['svo_shift_min']:.3f}, {update_info['svo_shift_max']:.3f}] std={update_info['svo_shift_std']:.3f}")
                print(f"  Target Scales : γV(s') = {update_info['abs_gamma_v_mean']:.3f} | |Q| = {update_info['abs_expert_q_mean']:.3f}")
                print(f"  SVO Impact    : |Shift| = {update_info['abs_svo_shift_mean']:.3f} | Ratio (|Shift|/|Q|) = {update_info['svo_shift_to_q_ratio']:.3f}")

                # Warnings for scaling issues
                if update_info['svo_shift_to_q_ratio'] > 1.0:
                    print("  >> WARNING: SVO shift is larger than Q-values. Consider lowering svo_lambda.")
                elif update_info['svo_shift_to_q_ratio'] < 0.001:
                    print("  >> WARNING: SVO shift is negligible. Consider increasing svo_lambda.")
            print()

            if use_wandb:
                import wandb
                wandb.log({
                    'eval/mean_reward':    eval_results['mean_reward'],
                    'eval/std_reward':     eval_results['std_reward'],
                    'eval/mean_length':    eval_results['mean_length'],
                    'eval/collision_rate': eval_results['collision_rate'],
                }, step=update)

        # Checkpoint
        if update % save_freq == 0:
            ckpt_path = os.path.join(output_dir, f'iq_learn_model_update_{update}.pt')
            trainer.save(ckpt_path)
            if use_wandb:
                import wandb
                wandb.save(ckpt_path, base_path=output_dir)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")

    print("Final evaluation (50 episodes)...")
    final_eval = trainer.evaluate(num_episodes=50)
    print(f"  Reward        : {final_eval['mean_reward']:.3f} ± {final_eval['std_reward']:.3f}")
    print(f"  Length        : {final_eval['mean_length']:.1f}")
    print(f"  Collision rate: {final_eval['collision_rate']:.2%}")

    if use_wandb:
        import wandb
        wandb.run.summary['final/mean_reward']    = final_eval['mean_reward']
        wandb.run.summary['final/std_reward']     = final_eval['std_reward']
        wandb.run.summary['final/mean_length']    = final_eval['mean_length']
        wandb.run.summary['final/collision_rate'] = final_eval['collision_rate']

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    final_model_path = os.path.join(output_dir, 'iq_learn_model_final.pt')
    trainer.save(final_model_path)

    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)

    plot_training_curves(trainer, output_dir)

    final_results = {
        'final_evaluation': final_eval,
        'expert_stats':     expert_stats,
        'training_config':  config_save,
    }
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    if use_wandb:
        import wandb
        wandb.save(final_model_path, base_path=output_dir)
        wandb.save(log_path, base_path=output_dir)
        wandb.log_artifact(final_model_path, name='iq_learn_final_model', type='model')
        wandb.finish()

    env.close()

    return trainer, training_log


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train IQ-Learn on highway-env')

    # Paths
    parser.add_argument('--expert-demo', type=str, required=True)
    parser.add_argument('--output-dir',  type=str, default=None)

    # Training
    parser.add_argument('--num-updates',  type=int,   default=10000)
    parser.add_argument('--batch-size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--gamma',        type=float, default=0.99)
    parser.add_argument('--temperature',  type=float, default=1.0)

    # IQ-Learn
    parser.add_argument('--method',            type=str,   default='value', choices=['value', 'q'])
    parser.add_argument('--loss-type',         type=str,   default='v0',    choices=['v0', 'v1'])
    parser.add_argument('--regularize-weight', type=float, default=1.0)

    # SVO regularization
    parser.add_argument('--svo-regularize', action='store_true',
                        help='Enable SVO regularization of the IQ-Learn objective.')
    parser.add_argument('--svo-alpha', type=float, default=0.0,
                        help='Target SVO angle in radians (only used when --svo-regularize is set).')
    parser.add_argument('--svo-lambda', type=float, default=1.0,
                        help='SVO regularization strength (only used when --svo-regularize is set).')
    parser.add_argument('--normalize-svo', action='store_true',
                        help='Batch-normalize R_SVO to zero mean / unit variance before applying.')

    # Learner rollouts
    parser.add_argument('--collect-learner-data', action='store_true')
    parser.add_argument('--learner-freq',  type=int, default=1000)
    parser.add_argument('--learner-steps', type=int, default=1000)

    # Evaluation
    parser.add_argument('--eval-freq',     type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=10)

    # W&B
    parser.add_argument('--wandb',          action='store_true')
    parser.add_argument('--wandb-project',  type=str, default='svo-irl')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb-tags',     type=str, nargs='*', default=[])

    # Misc
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        svo_label = f"_svo{np.degrees(args.svo_alpha):.0f}deg" if args.svo_regularize else ""
        args.output_dir = f'./iq_learn_runs/run{svo_label}_{timestamp}'

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else args.device


    train_iq_learn(
        env_config=ENV_CONFIG,
        expert_demo_path=args.expert_demo,
        output_dir=args.output_dir,
        num_updates=args.num_updates,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        temperature=args.temperature,
        method=args.method,
        loss_type=args.loss_type,
        regularize_weight=args.regularize_weight,
        # SVO
        use_svo=args.svo_regularize,
        svo_alpha=args.svo_alpha,
        svo_lambda=args.svo_lambda,
        normalize_svo=args.normalize_svo,
        # Training
        collect_learner_data=args.collect_learner_data,
        learner_collection_freq=args.learner_freq,
        learner_rollout_steps=args.learner_steps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    print(f"\n{'='*60}")
    print(f"Training complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()