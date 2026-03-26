"""
Evaluate three intersection experts with the counterfactual yielding wrapper
and analyze r_self / r_global separability.

Usage:
    python analyze_intersection_separability.py \
        --egoistic data/experts/expert_ego/final_model.zip \
        --prosocial data/experts/expert_pro/final_model.zip \
        --altruistic data/experts/expert_alt/final_model.zip \
        --episodes 100 \
        --target-transitions 500
"""

import argparse
import numpy as np
import gymnasium as gym
from itertools import combinations
from stable_baselines3 import DQN, PPO

from configs.intersection_config_new import INTERSECTION_CONFIG
from src.envs.intersection_yielding_wrapper import SVOYieldingWrapper


def load_model(path):
    try:
        return DQN.load(path)
    except Exception:
        return PPO.load(path)


def collect_transitions(model, env, num_episodes, max_transitions=None, seed=42):
    """Roll out a model and collect per-transition r_self, r_global."""
    transitions = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            transitions.append({
                'r_self': info['rewards/component_self'],
                'r_global': info['rewards/component_global'],
            })

            if max_transitions and len(transitions) >= max_transitions:
                return transitions

    return transitions


def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled < 1e-10:
        diff = a.mean() - b.mean()
        return np.sign(diff) * np.inf if abs(diff) > 1e-10 else 0.0
    return (a.mean() - b.mean()) / pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--egoistic', type=str, required=True)
    parser.add_argument('--prosocial', type=str, required=True)
    parser.add_argument('--altruistic', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--target-transitions', type=int, default=500,
                        help='Max transitions per mode (for balance)')
    parser.add_argument('--svo-angle', type=float, default=0.0,
                        help='SVO angle for the wrapper (0 = just collect r components)')
    args = parser.parse_args()

    # Create env with yielding wrapper
    config = INTERSECTION_CONFIG.copy()
    config['offscreen_rendering'] = True
    env = gym.make(config['id'], config=config)
    env.unwrapped.configure(config)
    env = SVOYieldingWrapper(env, svo_alpha=np.deg2rad(args.svo_angle))

    profiles = {
        'egoistic': args.egoistic,
        'prosocial': args.prosocial,
        'altruistic': args.altruistic,
    }

    print("=" * 65)
    print("Intersection-v1 — Counterfactual Yielding Separability Analysis")
    print("=" * 65)

    # ── Collect transitions ──
    all_data = {}
    for name, path in profiles.items():
        print(f"\nCollecting {name} from {path}...")
        model = load_model(path)
        trans = collect_transitions(model, env, args.episodes,
                                    max_transitions=args.target_transitions)
        all_data[name] = trans
        print(f"  {len(trans)} transitions collected")

    env.close()

    # ── Balance to equal counts ──
    min_count = min(len(t) for t in all_data.values())
    target = min(min_count, args.target_transitions)
    for name in all_data:
        all_data[name] = all_data[name][:target]
    print(f"\nBalanced to {target} transitions per mode")

    mode_names = list(profiles.keys())

    # ── Per-mode stats ──
    print(f"\n{'=' * 65}")
    print("TRANSITION-LEVEL COMPONENT STATS")
    print(f"{'=' * 65}")
    print(f"\n{'Mode':<14s} | {'r_self':>18s} | {'r_global':>18s} | {'n':>6s}")
    print(f"{'-'*14}-+-{'-'*18}-+-{'-'*18}-+-{'-'*6}")

    for mode in mode_names:
        rs = np.array([t['r_self'] for t in all_data[mode]])
        rg = np.array([t['r_global'] for t in all_data[mode]])
        print(f"  {mode:<12s} | {rs.mean():+7.3f} ± {rs.std():.3f}   | "
              f"{rg.mean():+7.3f} ± {rg.std():.3f}   | {len(rs):>6d}")

    # ── Pairwise Cohen's d ──
    print(f"\n{'=' * 65}")
    print("PAIRWISE COHEN'S d")
    print(f"{'=' * 65}")
    print(f"\n{'Pair':<30s} | {'d(r_self)':>10s} | {'d(r_global)':>12s} | {'max |d|':>8s}")
    print(f"{'-'*30}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

    for a, b in combinations(mode_names, 2):
        rs_a = np.array([t['r_self'] for t in all_data[a]])
        rs_b = np.array([t['r_self'] for t in all_data[b]])
        rg_a = np.array([t['r_global'] for t in all_data[a]])
        rg_b = np.array([t['r_global'] for t in all_data[b]])

        d_self = cohens_d(rs_a, rs_b)
        d_global = cohens_d(rg_a, rg_b)
        max_d = max(abs(d_self), abs(d_global))
        label = "LARGE" if max_d > 0.8 else ("MEDIUM" if max_d > 0.5 else "SMALL")

        print(f"  {a} vs {b:<15s} | {d_self:+8.3f}   | {d_global:+10.3f}   | {max_d:.3f} {label}")

    # ── R_SVO sweep ──
    print(f"\n{'=' * 65}")
    print("R_SVO SWEEP (transition-level)")
    print(f"{'=' * 65}")

    angles_deg = [0, 15, 30, 45, 60, 75, 90]

    header = f"{'α':>5s} |"
    for mode in mode_names:
        header += f" {mode:>10s} |"
    header += f" {'Δ(E,P)':>8s} | {'Δ(P,A)':>8s} | {'Δ(E,A)':>8s}"
    print(header)
    print("-" * len(header))

    for deg in angles_deg:
        alpha = np.deg2rad(deg)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)

        mode_svo = {}
        for mode in mode_names:
            svos = np.array([
                cos_a * t['r_self'] + sin_a * t['r_global']
                for t in all_data[mode]
            ])
            mode_svo[mode] = svos

        row = f"{deg:>4d}° |"
        for mode in mode_names:
            row += f" {mode_svo[mode].mean():+8.4f}   |"

        pairs = [('egoistic', 'prosocial'), ('prosocial', 'altruistic'),
                 ('egoistic', 'altruistic')]
        for a, b in pairs:
            d = cohens_d(mode_svo[a], mode_svo[b])
            row += f" {d:+7.2f}  |"
        print(row)

    # ── Verdict ──
    print(f"\n{'=' * 65}")
    print("SEPARABILITY VERDICT")
    print(f"{'=' * 65}")

    all_ok = True
    for a, b in combinations(mode_names, 2):
        rs_a = np.array([t['r_self'] for t in all_data[a]])
        rs_b = np.array([t['r_self'] for t in all_data[b]])
        rg_a = np.array([t['r_global'] for t in all_data[a]])
        rg_b = np.array([t['r_global'] for t in all_data[b]])
        max_d = max(abs(cohens_d(rs_a, rs_b)), abs(cohens_d(rg_a, rg_b)))
        label = "LARGE" if max_d > 0.8 else ("MEDIUM" if max_d > 0.5 else "SMALL")
        print(f"  {a} vs {b}: max |d| = {max_d:.3f} ({label})")
        if max_d < 0.5:
            all_ok = False

    if all_ok:
        print("\n  ✓ All pairwise separations are at least medium effect size.")
    else:
        print("\n  ✗ Some pairwise separations are weak.")
        print("    Consider adjusting reward weights or conflict detection parameters.")


if __name__ == "__main__":
    main()