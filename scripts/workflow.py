"""
Complete workflow: Extract demonstrations and train IQ-Learn.

Supports single-agent baselines and multi-agent mixed experiments.
SVO angles are always specified in RADIANS.

Common radian values for reference:
    0.0              ->   0 deg  (egoistic)
    0.3927 (pi/8)    ->  22.5 deg
    0.7854 (pi/4)    ->  45 deg
    1.5708 (pi/2)    ->  90 deg  (altruistic)
    5.4978 (7pi/4)   -> 315 deg  (competitive)

Use np.deg2rad(X) to convert if needed.
"""

import os
import math
import argparse
from datetime import datetime
import numpy as np
import gymnasium as gym
import highway_env
import torch

from scripts.extract_demonstrations import extract_from_multiple_agents, load_demonstrations
from scripts.train_iq_learn import train_iq_learn
# from src.envs.svo_pure_wrapper import SVOPureWrapper
from src.envs.intersection_yielding_wrapper import SVOYieldingWrapper as SVOPureWrapper

# ---------------------------------------------------------------------------
# Known agents: name -> svo_alpha in radians
# ---------------------------------------------------------------------------
KNOWN_AGENTS = {
    'egoistic':    0.0,
    'cooperative': math.pi / 8,
    'prosocial':   math.pi / 4,
    'altruistic':  math.pi / 2,
    'competitive': 7 * math.pi / 4,
}


def parse_svo_angle(value: str) -> float:
    """
    Parse SVO angle argument.

    Accepts:
        - A known agent name: 'egoistic', 'altruistic', etc.
        - A raw float in radians: '1.5708'
        - A degree shorthand:    'deg:90'

    Always returns a float in radians.
    """
    if value in KNOWN_AGENTS:
        return KNOWN_AGENTS[value]

    if value.startswith('deg:'):
        try:
            degrees = float(value[4:])
            return math.radians(degrees)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Could not parse degree value from '{value}'. "
                f"Expected format: 'deg:90'"
            )

    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid SVO angle. "
            f"Provide a float in radians, 'deg:X', or one of: "
            f"{list(KNOWN_AGENTS.keys())}"
        )


def create_env_fn(env_config, svo_alpha: float):
    def _create():
        env = gym.make(env_config['id'])
        env.unwrapped.config.update(env_config)
        env = SVOPureWrapper(env, svo_alpha=svo_alpha, lamb=1.0)
        return env
    return _create


def main():
    parser = argparse.ArgumentParser(
        description='Complete IQ-Learn workflow',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['extract', 'train', 'both'],
        help='extract: collect demos only\n'
             'train:   train IQ-Learn on existing demos\n'
             'both:    extract then train in one run',
    )

    # ------------------------------------------------------------------
    # Extraction arguments
    # ------------------------------------------------------------------
    parser.add_argument('--agent-path', type=str)
    parser.add_argument('--agent-name', type=str, default='expert')
    parser.add_argument(
        '--svo-angle', type=parse_svo_angle, default=0.0,
        metavar='RADIANS|deg:X|NAME',
        help=(
            'SVO angle for the environment wrapper (statistics only).\n'
            'Always stored and used in RADIANS.\n'
            'Accepted formats:\n'
            '  Raw radians:   1.5708\n'
            '  Degree helper: deg:90\n'
            '  Named preset:  egoistic | cooperative | prosocial |\n'
            '                 altruistic | competitive\n'
            'Default: 0.0  (egoistic baseline)'
        ),
    )
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--demo-save-dir', type=str,
                        default='./expert_demonstrations')

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    parser.add_argument('--demo-path', type=str)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-updates', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)

    # ------------------------------------------------------------------
    # IQ-Learn core (corrected to match train_iq_learn signature)
    # ------------------------------------------------------------------
    parser.add_argument('--loss-type', type=str, default='value',
                        choices=['value', 'value_expert', 'v0'],
                        help='Sampling strategy for 2nd loss term:\n'
                             '  value        – E_{all}[V(s)-γV(s\')] (online, default)\n'
                             '  value_expert – E_{expert}[V(s)-γV(s\')] (offline)\n'
                             '  v0           – (1-γ)E[V(s0)] (offline, usually suboptimal)')
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
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--use-target-network', action='store_true', default=True,
                        help='Use target network for V(s\') (default: True).')
    parser.add_argument('--grad-pen', action='store_true', default=False,
                        help='Enable gradient penalty (Wasserstein-1 proxy).')
    parser.add_argument('--lambda-gp', type=float, default=10.0,
                        help='Gradient penalty coefficient.')

    # ------------------------------------------------------------------
    # SVO regularization
    # ------------------------------------------------------------------
    parser.add_argument('--svo-regularize', action='store_true',
                        help='Enable SVO regularization of IQ-Learn.')
    parser.add_argument('--svo-mode', type=str, default='reward_reg',
                        choices=['bellman', 'reward_reg', 'reweight', 'reward_reg_reweight'],
                        help='SVO integration mode:\n'
                             '  bellman              – shift Bellman target\n'
                             '  reward_reg           – separate MSE on recovered reward\n'
                             '  reweight             – importance-weight expert sampling\n'
                             '  reward_reg_reweight  – both reward_reg + reweight combined')
    parser.add_argument('--svo-alpha-target', type=parse_svo_angle, default=0.0,
                        metavar='RADIANS|deg:X|NAME',
                        help='Target SVO angle for regularization (radians).')
    parser.add_argument('--svo-lambda', type=float, default=1.0,
                        help='SVO regularization strength λ.')
    parser.add_argument('--normalize-svo', action='store_true',
                        help='Batch-normalize R_SVO to zero mean / unit var.')
    parser.add_argument('--svo-reweight-temp', type=float, default=1.0,
                        help='Temperature for reweight mode softmax.')
    parser.add_argument('--svo-cumulative', action='store_true',
                        help='Use cumulative discounted SVO returns (G_self, G_global) '
                             'instead of instantaneous. Requires 10-element demo tuples.')

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='svo-irl')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=[])

    # ------------------------------------------------------------------
    # IQ-Learn Hyperparameters
    # ------------------------------------------------------------------
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)

    # ------------------------------------------------------------------
    # Online Learner Rollouts
    # ------------------------------------------------------------------
    parser.add_argument('--collect-learner-data', action='store_true')
    parser.add_argument('--learner-freq', type=int, default=10)
    parser.add_argument('--learner-steps', type=int, default=50)

    # ------------------------------------------------------------------
    # Evaluation & Saving
    # ------------------------------------------------------------------
    parser.add_argument('--eval-freq', type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--save-freq', type=int, default=500)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # from configs.env_config import ENV_CONFIG
    from configs.intersection_config_new import INTERSECTION_CONFIG as ENV_CONFIG

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\nSVO angle (env wrapper) : {args.svo_angle:.6f} rad  "
          f"({math.degrees(args.svo_angle):.2f} deg)")
    if args.svo_regularize:
        print(f"SVO target (regularizer): {args.svo_alpha_target:.6f} rad  "
              f"({math.degrees(args.svo_alpha_target):.2f} deg)")
        print(f"SVO mode                : {args.svo_mode}")
        print(f"SVO λ                   : {args.svo_lambda}")
        if args.svo_mode == 'reweight':
            print(f"Reweight temperature    : {args.svo_reweight_temp}")

    # ==================== EXTRACTION ====================
    if args.mode in ['extract', 'both']:
        print(f"\n{'='*60}")
        print("STEP 1: Extracting Expert Demonstrations")
        print(f"{'='*60}\n")

        if args.agent_path is None:
            raise ValueError("--agent-path is required for extraction")

        env_fn = create_env_fn(ENV_CONFIG, svo_alpha=args.svo_angle)
        agent_paths = {args.agent_name: args.agent_path}

        demo_paths = extract_from_multiple_agents(
            agent_paths=agent_paths,
            env_fn=env_fn,
            num_episodes_per_agent=args.num_episodes,
            save_dir=args.demo_save_dir,
            deterministic=True,
        )

        extracted_demo_path = demo_paths[args.agent_name]
        print(f"\n✓ Demonstrations extracted to: {extracted_demo_path}")

        if args.mode == 'both':
            args.demo_path = extracted_demo_path

    # ==================== TRAINING ====================
    if args.mode in ['train', 'both']:
        print(f"\n{'='*60}")
        print("STEP 2: Training IQ-Learn")
        print(f"{'='*60}\n")

        if args.demo_path is None:
            raise ValueError(
                "--demo-path is required for training (or use --mode both)")

        if args.output_dir is None:
            svo_label = ""
            if args.svo_regularize:
                svo_label = (f"_svo{math.degrees(args.svo_alpha_target):.0f}deg"
                             f"_{args.svo_mode}"
                             f"_lam{args.svo_lambda}")
            args.output_dir = (
                f'./iq_learn_runs/run_{args.agent_name}{svo_label}_{timestamp}'
            )

        wandb_run_name = args.wandb_run_name or f'{args.agent_name}_{timestamp}'

        trainer, training_log = train_iq_learn(
            env_config=ENV_CONFIG,
            expert_demo_path=args.demo_path,
            output_dir=args.output_dir,
            num_updates=args.num_updates,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            temperature=args.temperature,
            # IQ-Learn core (corrected)
            loss_type=args.loss_type,
            divergence=args.divergence,
            div_alpha=args.div_alpha,
            use_target_network=args.use_target_network,
            grad_pen=args.grad_pen,
            lambda_gp=args.lambda_gp,
            # SVO
            use_svo=args.svo_regularize,
            svo_mode=args.svo_mode,
            svo_alpha=args.svo_alpha_target,
            svo_lambda=args.svo_lambda,
            normalize_svo=args.normalize_svo,
            svo_reweight_temp=args.svo_reweight_temp,
            svo_cumulative=args.svo_cumulative,
            # Training
            collect_learner_data=args.collect_learner_data,
            learner_collection_freq=args.learner_freq,
            learner_rollout_steps=args.learner_steps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            seed=args.seed,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_tags=args.wandb_tags,
        )

        print(f"\n✓ Training complete! Results saved to: {args.output_dir}")

    print(f"\n{'='*60}")
    print("Workflow complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()