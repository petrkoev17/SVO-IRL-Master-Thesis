"""
Main training script for IQ-Learn with highway-env.

Updated to use the corrected IQ-Learn loss that faithfully implements
Garg et al. (2021) with all f-divergences, adapted for DQN.

Supports:
  - Standard IQ-Learn baseline (default: χ² divergence + 'value' sampling)
  - SVO-Regularised IQ-Learn with three modes:
      'bellman'    – shift Bellman target
      'reward_reg' – separate MSE loss on recovered reward (recommended)
      'reweight'   – importance-weight expert sampling
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
    env = gym.make(config['id'], config=config, render_mode=render_mode)
    env = SVOPureWrapper(env, svo_alpha=svo_angle, lamb=1.0)
    return env


def plot_training_curves(trainer: IQLearnTrainer, save_dir: str):
    """Plot and save training curves locally."""
    n_rows = 2
    if trainer.use_svo and trainer.svo_mode == 'reward_reg':
        n_rows = 3

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

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

    if n_rows == 3 and trainer.svo_ratios:
        axes[2].plot(trainer.svo_ratios, label='|SVO shift| / |Q|', alpha=0.7)
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Ratio')
        axes[2].set_title('SVO Regularisation Impact')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def train_iq_learn(
    env_config: Dict,
    expert_demo_path: str,
    output_dir: str,
    # IQ-Learn core
    num_updates: int = 10000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    temperature: float = 1.0,
    loss_type: str = 'value',
    divergence: str = 'chi',
    div_alpha: float = 0.5,
    use_target_network: bool = True,
    grad_pen: bool = False,
    lambda_gp: float = 10.0,
    # SVO
    use_svo: bool = False,
    svo_mode: str = 'reward_reg',
    svo_alpha: float = 0.0,
    svo_lambda: float = 1.0,
    normalize_svo: bool = False,
    svo_reweight_temp: float = 1.0,
    # Training
    collect_learner_data: bool = False,
    learner_collection_freq: int = 1000,
    learner_rollout_steps: int = 1000,
    eval_freq: int = 500,
    eval_episodes: int = 10,
    save_freq: int = 2000,
    device: str = 'cpu',
    seed: int = 42,
    # Wandb
    use_wandb: bool = False,
    wandb_project: str = 'svo-irl',
    wandb_run_name: str = None,
    wandb_tags: List[str] = None,
    # Legacy compatibility (accepted but unused by trainer)
    method: str = 'value',
    regularize_weight: float = 1.0,
):
    """Train IQ-Learn agent."""

    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_save = {
        'num_updates': num_updates,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'tau': tau,
        'temperature': temperature,
        'loss_type': loss_type,
        'divergence': divergence,
        'div_alpha': div_alpha,
        'use_target_network': use_target_network,
        'grad_pen': grad_pen,
        'lambda_gp': lambda_gp,
        'use_svo': use_svo,
        'svo_mode': svo_mode,
        'svo_alpha': svo_alpha,
        'svo_lambda': svo_lambda,
        'normalize_svo': normalize_svo,
        'svo_reweight_temp': svo_reweight_temp,
        'collect_learner_data': collect_learner_data,
        'seed': seed,
        'device': device,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=4)

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # W&B
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed. Run: pip install wandb")
            use_wandb = False

    if use_wandb:
        import wandb
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
    mode_label = {
        'bellman': 'Bellman-shift',
        'reward_reg': 'Reward-regularised',
        'reweight': 'Reweight-sampling',
        'reward_reg_reweight': 'Reward-reg + Reweight',
    }
    header = (f"SVO-IQ ({mode_label.get(svo_mode, svo_mode)})"
              if use_svo else "IQ-Learn (Baseline)")

    print(f"\n{'='*60}")
    print(f"{header} Training")
    print(f"{'='*60}")
    print(f"Output directory : {output_dir}")
    print(f"Device           : {device}")
    print(f"Expert demos     : {expert_demo_path}")
    print(f"Loss type        : {loss_type}")
    print(f"Divergence       : {divergence}")
    if divergence == 'chi':
        print(f"Div alpha (α)    : {div_alpha}")
    print(f"Use target net   : {use_target_network}")
    if use_svo:
        print(f"SVO mode         : {svo_mode}")
        print(f"SVO α_target     : {np.degrees(svo_alpha):.1f}° ({svo_alpha:.4f} rad)")
        print(f"SVO λ            : {svo_lambda}")
        print(f"SVO normalise    : {normalize_svo}")
        if svo_mode in ('reweight', 'reward_reg_reweight'):
            print(f"Reweight temp    : {svo_reweight_temp}")

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

    if expert_trajectories and expert_trajectories[0]:
        sample_len = len(expert_trajectories[0][0])
        has_svo = sample_len >= 8
        print(f"Transition format: {sample_len} elements "
              f"({'with' if has_svo else 'without'} SVO components)")
        if use_svo and not has_svo:
            print(
                "\n⚠  WARNING: SVO regularisation is enabled but demos "
                "lack r_self/r_global. Re-extract with updated script."
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
    print(f"\nInitialising trainer...")
    trainer = IQLearnTrainer(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        lr=learning_rate,
        gamma=gamma,
        tau=tau,
        device=device,
        # IQ-Learn core (corrected)
        loss_type=loss_type,
        divergence=divergence,
        div_alpha=div_alpha,
        temperature=temperature,
        use_target_network=use_target_network,
        grad_pen=grad_pen,
        lambda_gp=lambda_gp,
        # SVO
        use_svo=use_svo,
        svo_mode=svo_mode,
        svo_alpha=svo_alpha,
        svo_lambda=svo_lambda,
        normalize_svo=normalize_svo,
        svo_reweight_temp=svo_reweight_temp,
    )

    trainer.load_expert_demonstrations(expert_trajectories)

    if collect_learner_data:
        print("Prefilling learner with random exploration...")
        trainer.collect_learner_rollout(
            num_steps=2000,
            epsilon=1.0
        )

    # ------------------------------------------------------------------
    # Training log
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
        'softq_losses': [],
        'value_losses': [],
        'chi2_losses': [],
        # reward_reg mode
        'svo_reg_losses': [],
        'recovered_reward_means': [],
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
                num_steps=learner_rollout_steps, epsilon=epsilon)

        # Gradient update
        update_info = trainer.update(batch_size=batch_size)

        # --- Local log ---
        training_log['losses'].append(update_info['loss'])
        training_log['expert_q_means'].append(update_info['expert_q_mean'])
        training_log['abs_expert_q_means'].append(
            update_info.get('abs_expert_q_mean', 0.0))
        training_log['abs_gamma_v_means'].append(
            update_info.get('abs_gamma_v_mean', 0.0))
        training_log['softq_losses'].append(
            update_info.get('softq_loss', 0.0))
        training_log['value_losses'].append(
            update_info.get('value_loss', 0.0))
        training_log['chi2_losses'].append(
            update_info.get('chi2_loss', 0.0))

        if 'svo_raw_mean' in update_info:
            training_log['svo_reward_means'].append(update_info['svo_raw_mean'])
            training_log['abs_svo_shift_means'].append(
                update_info.get('abs_svo_shift_mean', 0.0))
            training_log['svo_shift_to_q_ratios'].append(
                update_info.get('svo_shift_to_q_ratio', 0.0))

        if 'svo_reg_loss' in update_info:
            training_log['svo_reg_losses'].append(update_info['svo_reg_loss'])
            training_log['recovered_reward_means'].append(
                update_info['recovered_reward_mean'])

        # --- W&B ---
        if use_wandb:
            import wandb
            log_dict = {
                'train/loss':              update_info['loss'],
                'train/expert_q_mean':     update_info['expert_q_mean'],
                'train/expert_v_mean':     update_info['expert_next_v_mean'],
                'train/abs_expert_q_mean': update_info.get('abs_expert_q_mean', 0.0),
                'train/abs_gamma_v_mean':  update_info.get('abs_gamma_v_mean', 0.0),
                'train/softq_loss':        update_info.get('softq_loss', 0.0),
            }
            if 'value_loss' in update_info:
                log_dict['train/value_loss'] = update_info['value_loss']
            if 'chi2_loss' in update_info:
                log_dict['train/chi2_loss'] = update_info['chi2_loss']
            if 'gp_loss' in update_info:
                log_dict['train/gp_loss'] = update_info['gp_loss']

            if 'learner_q_mean' in update_info:
                log_dict['train/learner_q_mean'] = update_info['learner_q_mean']
                log_dict['train/learner_v_mean'] = update_info['learner_next_v_mean']

            # SVO metrics
            if 'svo_raw_mean' in update_info:
                log_dict['train/svo_raw_mean'] = update_info['svo_raw_mean']
                log_dict['train/svo_raw_std']  = update_info.get('svo_raw_std', 0.0)

            if 'svo_shift_std' in update_info:
                log_dict['train/svo_shift_std'] = update_info['svo_shift_std']
                log_dict['train/svo_shift_min'] = update_info['svo_shift_min']
                log_dict['train/svo_shift_max'] = update_info['svo_shift_max']
                log_dict['train/abs_svo_shift_mean']   = update_info['abs_svo_shift_mean']
                log_dict['train/svo_shift_to_q_ratio'] = update_info['svo_shift_to_q_ratio']

            if 'svo_reg_loss' in update_info:
                log_dict['train/svo_reg_loss']          = update_info['svo_reg_loss']
                log_dict['train/recovered_reward_mean'] = update_info['recovered_reward_mean']
                log_dict['train/recovered_reward_std']  = update_info['recovered_reward_std']

            wandb.log(log_dict, step=update)

        # --- Progress bar ---
        if update % 10 == 0:
            desc = (f"Loss: {update_info['loss']:.4f}  "
                    f"Expert Q: {update_info['expert_q_mean']:.3f}")

            if 'chi2_loss' in update_info:
                desc += f"  χ²: {update_info['chi2_loss']:.4f}"

            if 'svo_reg_loss' in update_info:
                desc += f"  SVO reg: {update_info['svo_reg_loss']:.4f}"
                desc += f"  r̂ mean: {update_info['recovered_reward_mean']:.3f}"
            elif 'svo_raw_mean' in update_info:
                desc += f"  SVO raw: {update_info['svo_raw_mean']:.3f}"
                if 'svo_shift_min' in update_info:
                    desc += (f" shift:[{update_info['svo_shift_min']:.2f},"
                             f"{update_info['svo_shift_max']:.2f}]")
                if 'svo_shift_to_q_ratio' in update_info:
                    desc += f"  Ratio: {update_info['svo_shift_to_q_ratio']:.3f}"

            pbar.set_description(desc)

        # --- Periodic evaluation ---
        if update % eval_freq == 0:
            eval_results = trainer.evaluate(num_episodes=eval_episodes)

            training_log['updates'].append(update)
            training_log['eval_rewards'].append(eval_results['mean_reward'])
            training_log['eval_collision_rates'].append(eval_results['collision_rate'])

            print(f"\n[Update {update}/{num_updates}]")
            print(f"  Reward        : {eval_results['mean_reward']:.3f} "
                  f"± {eval_results['std_reward']:.3f}")
            print(f"  Length        : {eval_results['mean_length']:.1f}")
            print(f"  Collision rate: {eval_results['collision_rate']:.2%}")
            print(f"  Loss          : {update_info['loss']:.4f}")
            print(f"  Expert Q      : {update_info['expert_q_mean']:.3f}")
            if 'chi2_loss' in update_info:
                print(f"  χ² loss       : {update_info['chi2_loss']:.4f}")

            if 'svo_reg_loss' in update_info:
                print(f"  SVO reg loss  : {update_info['svo_reg_loss']:.4f}")
                print(f"  Recovered r̂   : {update_info['recovered_reward_mean']:.3f} "
                      f"± {update_info['recovered_reward_std']:.3f}")
                if 'svo_shift_to_q_ratio' in update_info:
                    print(f"  SVO Impact    : |Shift|/|Q| = "
                          f"{update_info['svo_shift_to_q_ratio']:.3f}")

            elif 'svo_raw_mean' in update_info:
                print(f"  SVO raw       : {update_info['svo_raw_mean']:.3f} "
                      f"± {update_info.get('svo_raw_std', 0):.3f}")
                if 'svo_shift_min' in update_info:
                    print(f"  SVO shift     : [{update_info['svo_shift_min']:.3f}, "
                          f"{update_info['svo_shift_max']:.3f}]")
                    print(f"  Target Scales : γV(s') = {update_info['abs_gamma_v_mean']:.3f} "
                          f"| |Q| = {update_info['abs_expert_q_mean']:.3f}")
                    print(f"  SVO Impact    : |Shift| = "
                          f"{update_info['abs_svo_shift_mean']:.3f} "
                          f"| Ratio = {update_info['svo_shift_to_q_ratio']:.3f}")

                    if update_info['svo_shift_to_q_ratio'] > 1.0:
                        print("  >> WARNING: SVO shift > Q-values. Lower svo_lambda.")
                    elif update_info['svo_shift_to_q_ratio'] < 0.001:
                        print("  >> WARNING: SVO shift negligible. Raise svo_lambda.")
            print()

            if use_wandb:
                import wandb
                wandb.log({
                    'eval/mean_reward':    eval_results['mean_reward'],
                    'eval/std_reward':     eval_results['std_reward'],
                    'eval/mean_length':    eval_results['mean_length'],
                    'eval/collision_rate': eval_results['collision_rate'],
                }, step=update)

        # --- Checkpoint ---
        if update % save_freq == 0:
            ckpt_path = os.path.join(output_dir,
                                     f'iq_learn_model_update_{update}.pt')
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
    print(f"  Reward        : {final_eval['mean_reward']:.3f} "
          f"± {final_eval['std_reward']:.3f}")
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


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train IQ-Learn on highway-env (corrected implementation)')

    # Paths
    parser.add_argument('--expert-demo', type=str, required=True)
    parser.add_argument('--output-dir',  type=str, default=None)

    # Training
    parser.add_argument('--num-updates',  type=int,   default=10000)
    parser.add_argument('--batch-size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--gamma',        type=float, default=0.99)
    parser.add_argument('--temperature',  type=float, default=1.0)

    # IQ-Learn core (corrected)
    parser.add_argument('--loss-type', type=str, default='value',
                        choices=['value', 'value_expert', 'v0'],
                        help='Sampling strategy for 2nd loss term:\n'
                             '  value        – E_{all}[V(s)-γV(s\')]  (online, default)\n'
                             '  value_expert – E_{expert}[V(s)-γV(s\')]  (offline)\n'
                             '  v0           – (1-γ)E[V(s0)]  (offline, usually suboptimal)')
    parser.add_argument('--divergence', type=str, default='chi',
                        choices=['chi', 'kl', 'kl2', 'kl_fix', 'js', 'hellinger', 'none'],
                        help='f-divergence for IQ-Learn objective:\n'
                             '  chi       – χ² divergence (recommended, default)\n'
                             '  kl        – KL (original dual, sub-optimal)\n'
                             '  kl2       – KL (biased dual)\n'
                             '  kl_fix    – KL (unbiased fix)\n'
                             '  js        – Jensen-Shannon\n'
                             '  hellinger – Hellinger\n'
                             '  none      – standard (no reweighting)')
    parser.add_argument('--div-alpha', type=float, default=0.5,
                        help='α parameter for χ² divergence regularisation. '
                             'Controls strength: 1/(4α) · E[r̂²]. '
                             'Only used when --divergence chi.')
    parser.add_argument('--use-target-network', action='store_true', default=True,
                        help='Use target network for V(s\') (default: True).')
    parser.add_argument('--no-target-network', dest='use_target_network',
                        action='store_false',
                        help='Disable target network for V(s\').')
    parser.add_argument('--grad-pen', action='store_true',
                        help='Enable gradient penalty (Q-magnitude penalty).')
    parser.add_argument('--lambda-gp', type=float, default=10.0,
                        help='Gradient penalty coefficient.')

    # SVO regularisation
    parser.add_argument('--svo-regularize', action='store_true',
                        help='Enable SVO regularisation.')
    parser.add_argument('--svo-mode', type=str, default='reward_reg',
                        choices=['bellman', 'reward_reg', 'reweight', 'reward_reg_reweight'],
                        help='How to integrate SVO into IQ-Learn.')
    parser.add_argument('--svo-alpha', type=float, default=0.0,
                        help='Target SVO angle in radians.')
    parser.add_argument('--svo-lambda', type=float, default=1.0,
                        help='SVO regularisation strength.')
    parser.add_argument('--normalize-svo', action='store_true',
                        help='Batch-normalise R_SVO to zero mean / unit variance.')
    parser.add_argument('--svo-reweight-temp', type=float, default=1.0,
                        help='Temperature for reweight mode softmax.')

    # Learner rollouts
    parser.add_argument('--collect-learner-data', action='store_true')
    parser.add_argument('--learner-freq',  type=int, default=1000)
    parser.add_argument('--learner-steps', type=int, default=1000)

    # Evaluation
    parser.add_argument('--eval-freq',     type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--save-freq',     type=int, default=2000)

    # W&B
    parser.add_argument('--wandb',          action='store_true')
    parser.add_argument('--wandb-project',  type=str, default='svo-irl')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb-tags',     type=str, nargs='*', default=[])

    # Misc
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')

    # Legacy compatibility (accepted but mapped to new params)
    parser.add_argument('--method', type=str, default=None,
                        help='DEPRECATED: use --loss-type instead.')
    parser.add_argument('--regularize-weight', type=float, default=1.0,
                        help='DEPRECATED: no longer used (χ² α controls regularisation).')

    args = parser.parse_args()

    # Handle legacy --method flag
    if args.method is not None:
        print(f"⚠  --method is deprecated. Mapping '{args.method}' -> --loss-type")
        if args.method == 'value':
            args.loss_type = 'value'
        elif args.method == 'q':
            args.loss_type = 'value_expert'
        else:
            print(f"   Unknown method '{args.method}', keeping --loss-type={args.loss_type}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        svo_label = ""
        if args.svo_regularize:
            svo_label = f"_svo{np.degrees(args.svo_alpha):.0f}deg_{args.svo_mode}"
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
        loss_type=args.loss_type,
        divergence=args.divergence,
        div_alpha=args.div_alpha,
        use_target_network=args.use_target_network,
        grad_pen=args.grad_pen,
        lambda_gp=args.lambda_gp,
        # SVO
        use_svo=args.svo_regularize,
        svo_mode=args.svo_mode,
        svo_alpha=args.svo_alpha,
        svo_lambda=args.svo_lambda,
        normalize_svo=args.normalize_svo,
        svo_reweight_temp=args.svo_reweight_temp,
        # Training
        collect_learner_data=args.collect_learner_data,
        learner_collection_freq=args.learner_freq,
        learner_rollout_steps=args.learner_steps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        device=device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()