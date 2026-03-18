"""
Mix & Analyze — Gridworld V2 Expert Demonstrations

1. Loads egoistic, prosocial, altruistic .pkl files
2. Combines them into a mixed dataset
3. Analyzes separability:
   - Per-mode r_self and r_global statistics
   - Pairwise Cohen's d on components
   - R_SVO at sweep of target angles: shows which angle best separates which mode
   - Trajectory-level atan2 recovered angle vs. ground truth
"""

import pickle
import numpy as np
from itertools import combinations


# ── Helpers ──────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d (positive = a > b). Returns inf if both have zero variance but different means."""
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled < 1e-10:
        diff = a.mean() - b.mean()
        if abs(diff) < 1e-10:
            return 0.0
        return np.sign(diff) * np.inf
    return (a.mean() - b.mean()) / pooled


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_transitions(trajectories, label):
    """Pull out per-transition r_self, r_global, and the mode label."""
    rows = []
    for traj in trajectories:
        for t in traj:
            # Format: (state, action, r, next_state, done, crashed, r_self, r_global)
            rows.append({
                'r_self': t[6],
                'r_global': t[7],
                'mode': label,
            })
    return rows


def extract_trajectory_stats(trajectories, label):
    """Per-trajectory cumulative stats."""
    rows = []
    for traj in trajectories:
        total_r_self = sum(t[6] for t in traj)
        total_r_global = sum(t[7] for t in traj)
        length = len(traj)
        rows.append({
            'total_r_self': total_r_self,
            'total_r_global': total_r_global,
            'mean_r_self': total_r_self / max(length, 1),
            'mean_r_global': total_r_global / max(length, 1),
            'length': length,
            'mode': label,
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Gridworld V2 — Mixed Dataset & Separability Analysis")
    print("=" * 65)

    # ── 1. Load ──
    profiles = {
        'egoistic':  './expert_demonstrations_v2/expert_egoistic.pkl',
        'prosocial': './expert_demonstrations_v2/expert_prosocial.pkl',
        'altruistic':'./expert_demonstrations_v2/expert_altruistic.pkl',
    }

    all_trajs = {}
    for name, path in profiles.items():
        all_trajs[name] = load_pkl(path)
        print(f"  Loaded {name}: {len(all_trajs[name])} trajectories, "
              f"{sum(len(t) for t in all_trajs[name])} transitions")

    # ── 2. Mix & Save ──
    mixed = []
    for name, trajs in all_trajs.items():
        mixed.extend(trajs)

    with open('./expert_demonstrations_v2/expert_mixed_all.pkl', 'wb') as f:
        pickle.dump(mixed, f)
    print(f"\n  Saved mixed dataset: {len(mixed)} trajectories → "
          f"expert_demonstrations_v2/expert_mixed_all.pkl")

    # ── 3. Per-mode component statistics ──
    print(f"\n{'=' * 65}")
    print("TRANSITION-LEVEL COMPONENT STATS")
    print(f"{'=' * 65}")

    trans_data = {}   # mode -> list of dicts
    for name, trajs in all_trajs.items():
        trans_data[name] = extract_transitions(trajs, name)

    mode_names = list(profiles.keys())

    print(f"\n{'Mode':<14s} | {'r_self':>18s} | {'r_global':>18s} | {'n':>6s}")
    print(f"{'-'*14}-+-{'-'*18}-+-{'-'*18}-+-{'-'*6}")
    for mode in mode_names:
        rs = np.array([t['r_self'] for t in trans_data[mode]])
        rg = np.array([t['r_global'] for t in trans_data[mode]])
        print(f"  {mode:<12s} | {rs.mean():+7.3f} ± {rs.std():.3f}   | "
              f"{rg.mean():+7.3f} ± {rg.std():.3f}   | {len(rs):>6d}")

    # ── 4. Trajectory-level stats ──
    print(f"\n{'=' * 65}")
    print("TRAJECTORY-LEVEL STATS")
    print(f"{'=' * 65}")

    traj_stats = {}
    for name, trajs in all_trajs.items():
        traj_stats[name] = extract_trajectory_stats(trajs, name)

    print(f"\n{'Mode':<14s} | {'Σr_self':>10s} | {'Σr_global':>10s} | "
          f"{'len':>5s} | {'atan2(Σg,Σs)':>12s}")
    print(f"{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}")
    for mode in mode_names:
        stats = traj_stats[mode]
        rs = np.array([s['total_r_self'] for s in stats])
        rg = np.array([s['total_r_global'] for s in stats])
        lens = np.array([s['length'] for s in stats])
        angles = np.rad2deg(np.arctan2(rg, rs))
        print(f"  {mode:<12s} | {rs.mean():+8.2f}  | {rg.mean():+8.2f}  | "
              f"{lens.mean():>5.1f} | {angles.mean():+7.1f}° ± {angles.std():.1f}°")

    # ── 5. Pairwise Cohen's d ──
    print(f"\n{'=' * 65}")
    print("PAIRWISE COHEN'S d (transition-level)")
    print(f"{'=' * 65}")

    print(f"\n{'Pair':<30s} | {'d(r_self)':>10s} | {'d(r_global)':>12s}")
    print(f"{'-'*30}-+-{'-'*10}-+-{'-'*12}")
    for a, b in combinations(mode_names, 2):
        rs_a = np.array([t['r_self'] for t in trans_data[a]])
        rs_b = np.array([t['r_self'] for t in trans_data[b]])
        rg_a = np.array([t['r_global'] for t in trans_data[a]])
        rg_b = np.array([t['r_global'] for t in trans_data[b]])
        d_self = cohens_d(rs_a, rs_b)
        d_global = cohens_d(rg_a, rg_b)
        print(f"  {a} vs {b:<15s} | {d_self:+8.3f}   | {d_global:+10.3f}")

    # ── 6. R_SVO sweep ──
    print(f"\n{'=' * 65}")
    print("R_SVO SEPARATION SWEEP (trajectory-level)")
    print(f"{'=' * 65}")
    print("\nFor each target angle α, compute R_SVO = cos(α)·Σr_self + sin(α)·Σr_global")
    print("per trajectory, then report mean R_SVO per mode and pairwise Cohen's d.\n")

    angles_deg = [0, 15, 30, 45, 60, 75, 90]

    # Header
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
            stats = traj_stats[mode]
            svos = np.array([
                cos_a * s['total_r_self'] + sin_a * s['total_r_global']
                for s in stats
            ])
            mode_svo[mode] = svos

        row = f"{deg:>4d}° |"
        for mode in mode_names:
            row += f" {mode_svo[mode].mean():+8.2f}   |"

        # Pairwise mean differences (more informative than d when std≈0)
        pairs = [('egoistic', 'prosocial'), ('prosocial', 'altruistic'),
                 ('egoistic', 'altruistic')]
        for a, b in pairs:
            diff = mode_svo[a].mean() - mode_svo[b].mean()
            row += f" {diff:+7.2f}  |"
        print(row)

    # ── 7. Recovered SVO angle ──
    print(f"\n{'=' * 65}")
    print("RECOVERED SVO ANGLE (atan2)")
    print(f"{'=' * 65}")
    print("\nGround truth vs recovered α = atan2(Σr_global, Σr_self)")
    print(f"\n{'Mode':<14s} | {'GT α':>6s} | {'Recovered α':>12s} | {'Error':>8s}")
    print(f"{'-'*14}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}")

    gt_angles = {'egoistic': 0.0, 'prosocial': 45.0, 'altruistic': 90.0}
    for mode in mode_names:
        stats = traj_stats[mode]
        rs = np.array([s['total_r_self'] for s in stats])
        rg = np.array([s['total_r_global'] for s in stats])
        recovered = np.rad2deg(np.arctan2(rg.mean(), rs.mean()))
        gt = gt_angles[mode]
        error = abs(recovered - gt)
        print(f"  {mode:<12s} | {gt:>5.0f}° | {recovered:>+9.1f}°    | {error:>6.1f}°")

    # ── 8. Verdict ──
    print(f"\n{'=' * 65}")
    print("SEPARABILITY VERDICT")
    print(f"{'=' * 65}")

    # Check if all pairwise d > 0.8 (large effect) at transition level
    all_large = True
    for a, b in combinations(mode_names, 2):
        rs_a = np.array([t['r_self'] for t in trans_data[a]])
        rs_b = np.array([t['r_self'] for t in trans_data[b]])
        rg_a = np.array([t['r_global'] for t in trans_data[a]])
        rg_b = np.array([t['r_global'] for t in trans_data[b]])
        d_self = abs(cohens_d(rs_a, rs_b))
        d_global = abs(cohens_d(rg_a, rg_b))
        max_d = max(d_self, d_global)
        label = "LARGE" if max_d > 0.8 else ("MEDIUM" if max_d > 0.5 else "SMALL")
        print(f"  {a} vs {b}: max |d| = {max_d:.3f} ({label})")
        if max_d < 0.5:
            all_large = False

    if all_large:
        print("\n  ✓ All pairwise separations are at least medium effect size.")
        print("    The reaction-based r_global produces separable SVO signals.")
    else:
        print("\n  ✗ Some pairwise separations are weak.")
        print("    Consider adjusting reward weights or reaction intensity.")


if __name__ == "__main__":
    main()