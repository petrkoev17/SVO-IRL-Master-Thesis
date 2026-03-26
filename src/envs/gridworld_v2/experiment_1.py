"""
Experiment 1 — Lambda Sweep (Behavioural Ambiguity)

Trains SVO-regularised IQ-Learn on a mixed dataset (egoistic + prosocial + altruistic)
with an EGOISTIC target (α=0°), sweeping λ from 0 (unregularised) to 5.

For each λ value:
  1. Train IQ-Learn with svo_lambda=λ, svo_alpha=0.0 (egoistic)
  2. Evaluate over multiple seeds
  3. Record: empirical SVO angle, mean r_self, mean r_global,
     success rate, mean episode length

Outputs:
  - Per-λ model checkpoints + training logs + eval logs
  - Summary JSON with all metrics
  - Summary plot (α̂ vs λ, success rate vs λ, etc.)

Usage:
    python sweep_lambda_exp1.py \
        --demo-path ./expert_demonstrations_v2/expert_mixed_all_v2.pkl \
        --updates 10000 \
        --output-dir ./runs/exp1_lambda_sweep
"""

import os
import sys
import math
import json
import pickle
import argparse
import numpy as np
import torch
from datetime import datetime

# ── Project imports ──────────────────────────────────────────────────
from src.envs.gridworld_v2.gridworld_v2 import SVOIntersectionGridV2
from src.algorithms.iq_learner import IQLearnTrainer


# ── Lambda values ────────────────────────────────────────────────────
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_policy(trainer, env, num_episodes=50, seeds=None):
    """
    Evaluate a trained policy across multiple seeds.
    Returns aggregated metrics and per-episode logs.
    """
    if seeds is None:
        seeds = [42, 100, 200, 300, 400]

    episode_logs = []
    all_r_self = []
    all_r_global = []
    all_rewards = []
    all_lengths = []
    all_arrivals = []

    for seed in seeds:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_r_self = 0.0
            ep_r_global = 0.0
            ep_reward = 0.0
            ep_len = 0
            arrived = False
            actions_taken = []

            while not done:
                action = trainer.select_action(obs, epsilon=0.0)
                actions_taken.append(int(action))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_r_self += info['rewards/component_self']
                ep_r_global += info['rewards/component_global']
                ep_reward += reward
                ep_len += 1
                if info.get('ego_arrived', False):
                    arrived = True

            all_r_self.append(ep_r_self)
            all_r_global.append(ep_r_global)
            all_rewards.append(ep_reward)
            all_lengths.append(ep_len)
            all_arrivals.append(float(arrived))

            episode_logs.append({
                'seed': seed,
                'episode': ep,
                'r_self': ep_r_self,
                'r_global': ep_r_global,
                'reward': ep_reward,
                'length': ep_len,
                'arrived': arrived,
                'actions': actions_taken,
            })

    mean_r_self = float(np.mean(all_r_self))
    mean_r_global = float(np.mean(all_r_global))

    # Empirical SVO angle (in degrees)
    empirical_alpha_rad = float(np.arctan2(mean_r_global, mean_r_self))
    empirical_alpha_deg = float(np.degrees(empirical_alpha_rad))

    summary = {
        'mean_r_self': mean_r_self,
        'std_r_self': float(np.std(all_r_self)),
        'mean_r_global': mean_r_global,
        'std_r_global': float(np.std(all_r_global)),
        'empirical_alpha_deg': empirical_alpha_deg,
        'empirical_alpha_rad': empirical_alpha_rad,
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'success_rate': float(np.mean(all_arrivals)),
        'mean_length': float(np.mean(all_lengths)),
        'std_length': float(np.std(all_lengths)),
        'num_episodes_total': len(episode_logs),
    }

    return summary, episode_logs


# ── Single training run ─────────────────────────────────────────────

def train_single_lambda(
    lam, demo_path, output_dir, updates, batch_size, lr, eval_freq, seed,
    eval_episodes, eval_seeds, svo_mode, hidden_dims,
):
    """Train one IQ-Learn agent with a specific lambda value."""
    run_dir = os.path.join(output_dir, f"lambda_{lam:.2f}_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Environment
    env = SVOIntersectionGridV2(deterministic_traffic=False)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    # Load demos (handle both dict and list formats)
    with open(demo_path, 'rb') as f:
        raw_data = pickle.load(f)

    if isinstance(raw_data, dict):
        expert_trajectories = raw_data['trajectories']
    else:
        expert_trajectories = raw_data

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine if SVO is enabled (λ=0 means unregularised baseline)
    use_svo = (lam > 0.0)

    # Trainer
    trainer = IQLearnTrainer(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        device=device,
        use_svo=use_svo,
        svo_mode=svo_mode,
        svo_alpha=0.0,          # Egoistic target
        svo_lambda=lam,
    )
    trainer.load_expert_demonstrations(expert_trajectories)

    # ── Training loop with full logging ──────────────────────────────
    training_log = {
        'updates': [],
        'losses': [],
        'expert_q_means': [],
        'eval_checkpoints': [],     # (update, eval_summary) pairs
    }

    # Add SVO-specific log keys if regularised
    if use_svo:
        training_log['svo_reg_losses'] = []
        training_log['recovered_reward_means'] = []
        training_log['svo_shift_to_q_ratios'] = []

    best_eval = None
    best_eval_logs = None

    for update in range(1, updates + 1):
        info = trainer.update(batch_size=batch_size)

        # Log training metrics every step
        training_log['updates'].append(update)
        training_log['losses'].append(float(info['loss']))
        training_log['expert_q_means'].append(float(info.get('expert_q_mean', 0.0)))

        if use_svo:
            training_log['svo_reg_losses'].append(
                float(info.get('svo_reg_loss', 0.0)))
            training_log['recovered_reward_means'].append(
                float(info.get('recovered_reward_mean', 0.0)))
            training_log['svo_shift_to_q_ratios'].append(
                float(info.get('svo_shift_to_q_ratio', 0.0)))

        # Periodic evaluation
        if update % eval_freq == 0:
            eval_summary, eval_ep_logs = evaluate_policy(
                trainer, env,
                num_episodes=eval_episodes,
                seeds=eval_seeds,
            )

            training_log['eval_checkpoints'].append({
                'update': update,
                'eval': eval_summary,
            })

            print(f"  [λ={lam:.2f} | seed={seed} | update {update}/{updates}] "
                  f"α̂={eval_summary['empirical_alpha_deg']:+.1f}° | "
                  f"r_self={eval_summary['mean_r_self']:.2f} | "
                  f"r_global={eval_summary['mean_r_global']:.2f} | "
                  f"success={eval_summary['success_rate']:.0%}")

            if best_eval is None or eval_summary['mean_reward'] > best_eval['mean_reward']:
                best_eval = eval_summary
                best_eval_logs = eval_ep_logs
                trainer.save(os.path.join(run_dir, 'best_model.pt'))

    # ── Final evaluation (more episodes) ─────────────────────────────
    final_eval, final_eval_logs = evaluate_policy(
        trainer, env,
        num_episodes=eval_episodes * 2,
        seeds=eval_seeds,
    )

    # ── Save everything ──────────────────────────────────────────────
    trainer.save(os.path.join(run_dir, 'final_model.pt'))

    # Training log
    with open(os.path.join(run_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    # Final eval episode-level logs
    with open(os.path.join(run_dir, 'final_eval_episodes.json'), 'w') as f:
        json.dump(final_eval_logs, f, indent=2)

    # Best eval episode-level logs
    if best_eval_logs is not None:
        with open(os.path.join(run_dir, 'best_eval_episodes.json'), 'w') as f:
            json.dump(best_eval_logs, f, indent=2)

    # Run summary
    run_results = {
        'lambda': lam,
        'seed': seed,
        'final_eval': final_eval,
        'best_eval': best_eval,
        'final_loss': float(np.mean(training_log['losses'][-100:])),
        'total_updates': updates,
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(run_results, f, indent=2)

    return run_results


# ── Convergence analysis ─────────────────────────────────────────────

def load_convergence_data(output_dir, lambdas, train_seeds):
    """
    Load eval checkpoints from all training logs.
    Returns: {lambda: {seed: [(update, alpha_deg), ...], ...}, ...}
    """
    convergence = {}
    for lam in lambdas:
        convergence[lam] = {}
        for seed in train_seeds:
            run_dir = os.path.join(output_dir, f"lambda_{lam:.2f}_seed_{seed}")
            log_path = os.path.join(run_dir, 'training_log.json')
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r') as f:
                log = json.load(f)

            curve = []
            for ckpt in log.get('eval_checkpoints', []):
                update = ckpt['update']
                alpha = ckpt['eval']['empirical_alpha_deg']
                curve.append((update, alpha))
            convergence[lam][seed] = curve

    return convergence


def compute_convergence_step(curve, target_deg=-42.0, threshold_deg=20.0):
    """
    Find the first eval checkpoint where α̂ is within threshold of target
    and stays there for all subsequent checkpoints.
    Returns the update number, or None if never converged.
    """
    for i, (update, alpha) in enumerate(curve):
        if abs(alpha - target_deg) <= threshold_deg:
            # Check it stays within threshold
            stayed = all(abs(a - target_deg) <= threshold_deg for _, a in curve[i:])
            if stayed:
                return update
    return None


def plot_convergence(convergence, output_dir, target_deg=-42.0, convergence_threshold=20.0):
    """
    Plot 1: α̂ vs training update — one line per λ, mean ± std over seeds.
    Plot 2: Convergence step vs λ.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not available — skipping convergence plots.")
        return

    lambdas = sorted(convergence.keys())

    # Skip λ=0 in the convergence curves (it's the unregularised baseline
    # that never converges to egoistic — show it as a horizontal reference)
    reg_lambdas = [l for l in lambdas if l > 0]

    # ── Plot 1: α̂ over training ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    cmap = cm.get_cmap('viridis', len(reg_lambdas))

    # Baseline reference: λ=0
    if 0.0 in convergence and convergence[0.0]:
        seeds_data = convergence[0.0]
        all_curves = list(seeds_data.values())
        if all_curves:
            updates = [u for u, _ in all_curves[0]]
            alphas = np.array([[a for _, a in c] for c in all_curves])
            mean_alpha = alphas.mean(axis=0)
            ax.axhline(y=mean_alpha[-1], color='gray', linestyle=':',
                        linewidth=1.5, label=f'λ=0 (baseline)', alpha=0.7)

    for i, lam in enumerate(reg_lambdas):
        seeds_data = convergence[lam]
        all_curves = list(seeds_data.values())
        if not all_curves:
            continue

        # Align by update step (all runs should have same eval_freq)
        updates = [u for u, _ in all_curves[0]]
        alphas = np.array([[a for _, a in c] for c in all_curves])

        mean_alpha = alphas.mean(axis=0)
        std_alpha = alphas.std(axis=0)

        color = cmap(i)
        ax.plot(updates, mean_alpha, color=color, linewidth=1.8,
                label=f'λ={lam}', marker='o', markersize=3)
        ax.fill_between(updates, mean_alpha - std_alpha,
                         mean_alpha + std_alpha, color=color, alpha=0.15)

    ax.axhline(y=target_deg, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                label=f'Target ({target_deg}°)')
    ax.axhline(y=target_deg + convergence_threshold, color='orange', linestyle='--',
                alpha=0.4, linewidth=1)
    ax.axhline(y=target_deg - convergence_threshold, color='orange', linestyle='--',
                alpha=0.4, linewidth=1)

    ax.set_xlabel('Training Update', fontsize=11)
    ax.set_ylabel('Empirical SVO Angle α̂ (°)', fontsize=11)
    ax.set_title('Convergence of α̂ During Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Convergence step vs λ ────────────────────────────────
    ax2 = axes[1]

    conv_steps_mean = []
    conv_steps_std = []
    conv_lambdas = []

    for lam in reg_lambdas:
        seeds_data = convergence[lam]
        steps = []
        for seed, curve in seeds_data.items():
            step = compute_convergence_step(curve, target_deg, convergence_threshold)
            if step is not None:
                steps.append(step)

        if steps:
            conv_lambdas.append(lam)
            conv_steps_mean.append(np.mean(steps))
            conv_steps_std.append(np.std(steps))

    if conv_lambdas:
        conv_steps_mean = np.array(conv_steps_mean)
        conv_steps_std = np.array(conv_steps_std)

        ax2.errorbar(conv_lambdas, conv_steps_mean, yerr=conv_steps_std,
                      marker='s', capsize=4, linewidth=2, markersize=7,
                      color='#2E86C1')
        ax2.set_xlabel('λ (SVO regularization strength)', fontsize=11)
        ax2.set_ylabel(f'Update to converge (|α̂ − {target_deg}°| < {convergence_threshold}°)',
                        fontsize=11)
        ax2.set_title('Convergence Speed vs λ', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Mark lambdas that never converged (any seed)
        for lam in reg_lambdas:
            if lam not in conv_lambdas:
                ax2.axvline(x=lam, color='red', linestyle=':', alpha=0.3)
                ax2.text(lam, ax2.get_ylim()[1] * 0.95, '✗',
                          ha='center', fontsize=12, color='red')
    else:
        ax2.text(0.5, 0.5, f'No runs converged to |α̂ − {target_deg}°| < {convergence_threshold}°',
                  ha='center', va='center', transform=ax2.transAxes, fontsize=11)

    plt.suptitle(
        'Experiment 1: Convergence Analysis — Egoistic Target on Mixed Data',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=200)
    plt.close()
    print(f"Convergence plot saved to {output_dir}/convergence_analysis.png")

    # ── Print convergence summary ────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"CONVERGENCE SUMMARY (target: {target_deg}°, threshold: ±{convergence_threshold}°)")
    print(f"{'=' * 65}")
    print(f"{'λ':>6s} | {'mean step':>10s} | {'std':>8s} | {'converged':>10s}")
    print(f"{'─' * 45}")

    for lam in reg_lambdas:
        seeds_data = convergence[lam]
        steps = []
        total = len(seeds_data)
        for seed, curve in seeds_data.items():
            step = compute_convergence_step(curve, target_deg, convergence_threshold)
            if step is not None:
                steps.append(step)

        if steps:
            print(f"{lam:6.2f} | {np.mean(steps):>8.0f}   | {np.std(steps):>6.0f}   | "
                  f"{len(steps)}/{total}")
        else:
            print(f"{lam:6.2f} | {'—':>10s} | {'—':>8s} | 0/{total}")

    print(f"{'=' * 65}")


# ── Plotting ─────────────────────────────────────────────────────────

def plot_sweep_results(all_results, output_dir):
    """Generate summary plots for the lambda sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    # Group by lambda, average over seeds
    from collections import defaultdict
    by_lambda = defaultdict(list)
    for r in all_results:
        by_lambda[r['lambda']].append(r['final_eval'])

    lambdas = sorted(by_lambda.keys())
    metrics = {
        'empirical_alpha_deg': ('Empirical SVO Angle (°)', 'α̂'),
        'mean_r_self': ('Mean Σr_self', 'r_self'),
        'mean_r_global': ('Mean Σr_global', 'r_global'),
        'success_rate': ('Success Rate', 'success'),
        'mean_reward': ('Mean Reward', 'reward'),
        'mean_length': ('Mean Episode Length', 'length'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (key, (title, short)) in enumerate(metrics.items()):
        ax = axes[idx]
        means = []
        stds = []
        for lam in lambdas:
            vals = [e[key] for e in by_lambda[lam]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(lambdas, means, yerr=stds, marker='o', capsize=4,
                     linewidth=2, markersize=6)
        ax.set_xlabel('λ (SVO regularization strength)', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add reference lines for SVO angle plot
        if key == 'empirical_alpha_deg':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5,
                        label='Egoistic (0°)')
            ax.axhline(y=45, color='green', linestyle='--', alpha=0.5,
                        label='Prosocial (45°)')
            ax.axhline(y=90, color='blue', linestyle='--', alpha=0.5,
                        label='Altruistic (90°)')
            ax.legend(fontsize=8)

    plt.suptitle(
        'Experiment 1: Lambda Sweep — Egoistic Target on Mixed Data\n'
        '(equal parts egoistic + prosocial + altruistic demonstrations)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_sweep_summary.png'), dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir}/lambda_sweep_summary.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Lambda sweep for behavioural ambiguity')

    parser.add_argument('--demo-path', type=str, required=True,
                        help='Path to the mixed expert .pkl file')
    parser.add_argument('--output-dir', type=str,
                        default='./runs/exp1_lambda_sweep',
                        help='Root output directory')

    # Training
    parser.add_argument('--updates', type=int, default=10000,
                        help='Training updates per run')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-freq', type=int, default=1000,
                        help='Evaluate every N updates')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer sizes')

    # SVO
    parser.add_argument('--svo-mode', type=str, default='reward_reg',
                        choices=['bellman', 'reward_reg', 'reweight',
                                 'reward_reg_reweight'])

    # Evaluation
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Episodes per seed during training eval')
    parser.add_argument('--eval-seeds', type=int, nargs='+',
                        default=[42, 100, 200, 300, 400],
                        help='Seeds for evaluation')

    # Reproducibility
    parser.add_argument('--train-seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 1024],
                        help='Training seeds (runs per lambda)')

    # Lambdas (override default if needed)
    parser.add_argument('--lambdas', type=float, nargs='+',
                        default=LAMBDA_VALUES,
                        help='Lambda values to sweep')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = timestamp
    config['lambda_values'] = args.lambdas
    with open(os.path.join(output_dir, 'sweep_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("EXPERIMENT 1: Lambda Sweep — Behavioural Ambiguity")
    print("=" * 70)
    print(f"Target SVO:       Egoistic (α = 0°)")
    print(f"Lambda values:    {args.lambdas}")
    print(f"Training seeds:   {args.train_seeds}")
    print(f"Updates per run:  {args.updates}")
    print(f"SVO mode:         {args.svo_mode}")
    print(f"Demo path:        {args.demo_path}")
    print(f"Output:           {output_dir}")
    print(f"Total runs:       {len(args.lambdas)} × {len(args.train_seeds)} "
          f"= {len(args.lambdas) * len(args.train_seeds)}")
    print("=" * 70)

    # ── Run sweep ────────────────────────────────────────────────────
    all_results = []

    for lam_idx, lam in enumerate(args.lambdas):
        print(f"\n{'─' * 60}")
        print(f"  λ = {lam:.2f}  ({lam_idx + 1}/{len(args.lambdas)})")
        print(f"{'─' * 60}")

        for seed in args.train_seeds:
            result = train_single_lambda(
                lam=lam,
                demo_path=args.demo_path,
                output_dir=output_dir,
                updates=args.updates,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_freq=args.eval_freq,
                seed=seed,
                eval_episodes=args.eval_episodes,
                eval_seeds=args.eval_seeds,
                svo_mode=args.svo_mode,
                hidden_dims=args.hidden_dims,
            )
            all_results.append(result)

    # ── Save aggregate results ───────────────────────────────────────
    summary_path = os.path.join(output_dir, 'sweep_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {summary_path}")

    # ── Print summary table ──────────────────────────────────────────
    from collections import defaultdict
    by_lambda = defaultdict(list)
    for r in all_results:
        by_lambda[r['lambda']].append(r['final_eval'])

    print(f"\n{'=' * 75}")
    print(f"{'λ':>6s} | {'α̂ (°)':>12s} | {'r_self':>10s} | {'r_global':>10s} | "
          f"{'success':>10s}")
    print(f"{'─' * 75}")

    for lam in sorted(by_lambda.keys()):
        evals = by_lambda[lam]
        alpha_mean = np.mean([e['empirical_alpha_deg'] for e in evals])
        alpha_std = np.std([e['empirical_alpha_deg'] for e in evals])
        rs_mean = np.mean([e['mean_r_self'] for e in evals])
        rg_mean = np.mean([e['mean_r_global'] for e in evals])
        suc_mean = np.mean([e['success_rate'] for e in evals])

        print(f"{lam:6.2f} | {alpha_mean:+6.1f}±{alpha_std:4.1f}  | "
              f"{rs_mean:+8.2f}   | {rg_mean:+8.2f}   | "
              f"{suc_mean:8.0%}")

    print(f"{'=' * 75}")

    # ── Plot ─────────────────────────────────────────────────────────
    plot_sweep_results(all_results, output_dir)

    # ── Convergence analysis ─────────────────────────────────────────
    convergence = load_convergence_data(output_dir, args.lambdas, args.train_seeds)
    plot_convergence(convergence, output_dir, target_deg=-42.0, convergence_threshold=20.0)

    print(f"\n✓ Experiment 1 complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()