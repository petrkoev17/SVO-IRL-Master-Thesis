"""
Dataset Discriminability Analysis Tool

Analyzes any demonstration .pkl file to show how well R_SVO separates
the constituent expert modes at different target angles.

Produces:
  - Component-level stats (r_self, r_global) per mode
  - Pairwise Cohen's d between all modes
  - R_SVO separation at all angles (transition-level and trajectory-level)
  - Target-mode-vs-rest analysis (the actual IRL use case)
  - Classification accuracy at each angle
  - Summary verdict: which angles are viable for mode selection

Usage:
    python analyze_discriminability.py --pkl path/to/dataset.pkl
    python analyze_discriminability.py --pkl path/to/dataset.pkl --plot
    python analyze_discriminability.py --pkl path/to/dataset.pkl --output results.txt

The script auto-detects the number of modes from metadata (source_paths + weights)
or falls back to a user-specified split.
"""

import pickle
import numpy as np
import argparse
import sys
import os
from itertools import combinations


def cohens_d(a, b):
    """Compute Cohen's d (positive = a scores higher)."""
    ps = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / (ps + 1e-8)


def classification_accuracy(scores_a, scores_b):
    """Median-threshold classification accuracy between two groups."""
    all_scores = np.concatenate([scores_a, scores_b])
    median = np.median(all_scores)
    correct_a = (scores_a > median).sum()
    correct_b = (scores_b <= median).sum()
    acc = (correct_a + correct_b) / (len(scores_a) + len(scores_b))
    return max(acc, 1 - acc)  # handle sign-agnostic accuracy


def load_dataset(pkl_path):
    """Load a demonstration .pkl file and extract transitions with mode labels."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        trajs = data['trajectories']
        metadata = data.get('metadata', {})
        stats = data.get('stats', {})
    elif isinstance(data, list):
        trajs = data
        metadata = {}
        stats = {}
    else:
        raise ValueError(f"Unknown data format: {type(data)}")

    return trajs, metadata, stats


def infer_mode_labels(trajs, metadata):
    """
    Assign a mode label to each trajectory based on metadata.

    Uses source_paths and weights to determine the split.
    Falls back to naming by index if metadata is unavailable.
    """
    n_trajs = len(trajs)
    source_paths = metadata.get('source_paths', [])
    weights = metadata.get('weights', [])

    if source_paths and weights:
        # Compute trajectory counts per source
        counts = [int(round(w * n_trajs)) for w in weights]
        # Adjust rounding to match total
        diff = n_trajs - sum(counts)
        if diff != 0:
            counts[-1] += diff

        # Extract mode names from file paths
        mode_names = []
        for path in source_paths:
            basename = os.path.basename(path).replace('.pkl', '')
            # Try to extract angle from common naming patterns
            name = basename
            for prefix in ['final_model_', 'expert_', 'demonstrations_']:
                name = name.replace(prefix, '')
            mode_names.append(name)

        # Assign labels
        labels = []
        for mode_name, count in zip(mode_names, counts):
            labels.extend([mode_name] * count)

        # Pad or trim if needed
        while len(labels) < n_trajs:
            labels.append(mode_names[-1])
        labels = labels[:n_trajs]

        return labels, mode_names
    else:
        # No metadata — treat entire dataset as one mode
        labels = ['unknown'] * n_trajs
        return labels, ['unknown']


def extract_transition_data(trajs, traj_labels):
    """Extract r_self, r_global, and mode labels per transition."""
    r_selfs, r_globals = [], []
    G_selfs, G_globals = [], []
    modes = []
    traj_ids = []

    has_cumulative = len(trajs[0][0]) >= 10 if trajs and trajs[0] else False

    for ti, traj in enumerate(trajs):
        mode = traj_labels[ti]
        for t in traj:
            r_selfs.append(float(t[6]))
            r_globals.append(float(t[7]))
            if has_cumulative:
                G_selfs.append(float(t[8]))
                G_globals.append(float(t[9]))
            modes.append(mode)
            traj_ids.append(ti)

    result = {
        'r_self': np.array(r_selfs),
        'r_global': np.array(r_globals),
        'mode': np.array(modes),
        'traj_id': np.array(traj_ids),
        'has_cumulative': has_cumulative,
    }

    if has_cumulative:
        result['G_self'] = np.array(G_selfs)
        result['G_global'] = np.array(G_globals)

    return result


def extract_trajectory_data(trajs, traj_labels):
    """Extract trajectory-level aggregations."""
    has_cumulative = len(trajs[0][0]) >= 10 if trajs and trajs[0] else False

    traj_data = {
        'mean_r_self': [],
        'mean_r_global': [],
        'mode': [],
    }

    if has_cumulative:
        traj_data['G_self_t0'] = []
        traj_data['G_global_t0'] = []

    for ti, traj in enumerate(trajs):
        traj_data['mean_r_self'].append(np.mean([t[6] for t in traj]))
        traj_data['mean_r_global'].append(np.mean([t[7] for t in traj]))
        traj_data['mode'].append(traj_labels[ti])
        if has_cumulative:
            traj_data['G_self_t0'].append(float(traj[0][8]))
            traj_data['G_global_t0'].append(float(traj[0][9]))

    for k in traj_data:
        traj_data[k] = np.array(traj_data[k])

    return traj_data


def print_section(title, file=None):
    """Print a section header."""
    line = "=" * 75
    print(f"\n{line}", file=file)
    print(f"{title}", file=file)
    print(f"{line}", file=file)


def analyze(pkl_path, do_plot=False, output_file=None):
    """Run full discriminability analysis."""
    out = open(output_file, 'w') if output_file else None

    def p(*args, **kwargs):
        print(*args, **kwargs)
        if out:
            print(*args, **kwargs, file=out)

    # ---- Load ----
    trajs, metadata, stats = load_dataset(pkl_path)
    traj_labels, mode_names = infer_mode_labels(trajs, metadata)

    p(f"Dataset: {pkl_path}")
    p(f"Trajectories: {len(trajs)}")
    p(f"Total transitions: {sum(len(t) for t in trajs)}")
    p(f"Modes detected: {mode_names}")
    if metadata.get('weights'):
        p(f"Weights: {metadata['weights']}")
    if metadata.get('cumulative_svo'):
        p(f"Has cumulative SVO: {metadata['cumulative_svo']}")
    p(f"Transition tuple length: {len(trajs[0][0])}")

    mode_counts = {m: sum(1 for l in traj_labels if l == m) for m in mode_names}
    for m, c in mode_counts.items():
        p(f"  {m}: {c} trajectories")

    if len(mode_names) < 2:
        p("\n⚠ Only one mode detected. Cannot compute separation.")
        if out:
            out.close()
        return

    # ---- Extract ----
    trans = extract_transition_data(trajs, traj_labels)
    traj_data = extract_trajectory_data(trajs, traj_labels)

    # ---- Component stats per mode ----
    print_section("COMPONENT STATS PER MODE (instantaneous)")
    p(f"\n{'Mode':<25s} | {'r_self':>20s} | {'r_global':>20s} | {'n_trans':>8s}")
    p(f"{'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*8}")

    for mode in mode_names:
        mask = trans['mode'] == mode
        rs = trans['r_self'][mask]
        rg = trans['r_global'][mask]
        p(f"  {mode:<23s} | {rs.mean():+8.4f} ± {rs.std():.4f}   | "
          f"{rg.mean():+8.4f} ± {rg.std():.4f}   | {mask.sum():>8d}")

    if trans['has_cumulative']:
        print_section("COMPONENT STATS PER MODE (cumulative)")
        p(f"\n{'Mode':<25s} | {'G_self':>20s} | {'G_global':>20s}")
        p(f"{'-'*25}-+-{'-'*20}-+-{'-'*20}")
        for mode in mode_names:
            mask = trans['mode'] == mode
            gs = trans['G_self'][mask]
            gg = trans['G_global'][mask]
            p(f"  {mode:<23s} | {gs.mean():+8.4f} ± {gs.std():.4f}   | "
              f"{gg.mean():+8.4f} ± {gg.std():.4f}  ")

    # ---- Pairwise Cohen's d on components ----
    print_section("PAIRWISE COHEN'S d — COMPONENTS")

    components = [('r_self', trans['r_self']), ('r_global', trans['r_global'])]
    if trans['has_cumulative']:
        components += [('G_self', trans['G_self']), ('G_global', trans['G_global'])]

    p(f"\n{'Pair':<40s} | " + " ".join(f"{name:>10s}" for name, _ in components))
    p(f"{'-'*40}-+-" + "-".join(["-" * 10] * len(components)))

    for a, b in combinations(mode_names, 2):
        mask_a = trans['mode'] == a
        mask_b = trans['mode'] == b
        ds = []
        for comp_name, comp_vals in components:
            d = cohens_d(comp_vals[mask_a], comp_vals[mask_b])
            ds.append(d)
        row = " ".join(f"{d:+10.4f}" for d in ds)
        p(f"  {a} vs {b:<24s} | {row}")

    # ---- R_SVO across angles — transition level ----
    angles = [0, 15, 22.5, 30, 45, 60, 67.5, 75, 90, 135, 180, 225, 270, 315]

    print_section("R_SVO — TRANSITION-LEVEL — ALL PAIRWISE")
    p(f"\n  Positive d = first mode scores higher on R_SVO")

    for a_name, b_name in combinations(mode_names, 2):
        mask_a = trans['mode'] == a_name
        mask_b = trans['mode'] == b_name

        p(f"\n  {a_name} vs {b_name}:")
        p(f"  {'Angle':>7s} | {'d':>8s} | {'acc':>6s} | {'effect':>12s}")
        p(f"  {'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*12}")

        for angle_deg in angles:
            rad = np.radians(angle_deg)
            rsvo = np.cos(rad) * trans['r_self'] + np.sin(rad) * trans['r_global']
            d = cohens_d(rsvo[mask_a], rsvo[mask_b])
            acc = classification_accuracy(rsvo[mask_a], rsvo[mask_b])

            if abs(d) > 0.8:
                effect = "LARGE ✓"
            elif abs(d) > 0.5:
                effect = "medium"
            elif abs(d) > 0.2:
                effect = "small"
            else:
                effect = "negligible ✗"

            p(f"  {angle_deg:7.1f} | {d:+8.4f} | {acc:5.1%} | {effect}")

    # ---- Target mode vs rest (the IRL use case) ----
    print_section("TARGET MODE vs REST OF DATASET (IRL use case)")
    p(f"\n  This is what reward_reg / reweight actually needs:")
    p(f"  d > 0 means the target mode scores HIGHER than everything else\n")

    p(f"  {'Angle':>7s} |", end="")
    for mode in mode_names:
        p(f" {mode:>14s}", end="")
    p()
    p(f"  {'-'*7}-+-" + "-".join(["-" * 14] * len(mode_names)))

    for angle_deg in angles:
        rad = np.radians(angle_deg)
        rsvo = np.cos(rad) * trans['r_self'] + np.sin(rad) * trans['r_global']

        p(f"  {angle_deg:7.1f} |", end="")
        for mode in mode_names:
            target_mask = trans['mode'] == mode
            rest_mask = ~target_mask
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            marker = "✓" if abs(d) > 0.8 else " "
            p(f" {d:+7.3f} {marker:>5s}", end="")
        p()

    # ---- Trajectory level ----
    print_section("TRAJECTORY-LEVEL ANALYSIS (mean per trajectory)")

    p(f"\n  {'Angle':>7s} |", end="")
    for mode in mode_names:
        p(f" {mode:>14s}", end="")
    p()
    p(f"  {'-'*7}-+-" + "-".join(["-" * 14] * len(mode_names)))

    for angle_deg in [0, 22.5, 45, 67.5, 90, 270, 315]:
        rad = np.radians(angle_deg)
        rsvo = np.cos(rad) * traj_data['mean_r_self'].astype(float) + \
               np.sin(rad) * traj_data['mean_r_global'].astype(float)

        p(f"  {angle_deg:7.1f} |", end="")
        for mode in mode_names:
            target_mask = traj_data['mode'] == mode
            rest_mask = ~target_mask
            if target_mask.sum() == 0 or rest_mask.sum() == 0:
                p(f" {'N/A':>14s}", end="")
                continue
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            acc = classification_accuracy(rsvo[target_mask], rsvo[rest_mask])
            p(f" {d:+5.2f} {acc:5.1%}", end="")
        p()

    # ---- Best angle per target ----
    print_section("BEST ANGLE FOR EACH TARGET MODE")

    fine_angles = np.arange(0, 360, 2.5)

    for mode in mode_names:
        target_mask = trans['mode'] == mode
        rest_mask = ~target_mask

        best_d = -float('inf')
        best_angle = 0

        for angle_deg in fine_angles:
            rad = np.radians(angle_deg)
            rsvo = np.cos(rad) * trans['r_self'] + np.sin(rad) * trans['r_global']
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            if d > best_d:
                best_d = d
                best_angle = angle_deg

        p(f"\n  {mode}:")
        p(f"    Best angle: {best_angle:5.1f}°  d = {best_d:+.4f}  "
          f"{'✓ VIABLE' if abs(best_d) > 0.8 else '✗ WEAK'}")

        # Also show d at a few standard angles
        for std_angle in [0, 45, 90, 315]:
            rad = np.radians(std_angle)
            rsvo = np.cos(rad) * trans['r_self'] + np.sin(rad) * trans['r_global']
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            p(f"    α = {std_angle:5.1f}°:  d = {d:+.4f}")

    # ---- Summary verdict ----
    print_section("SUMMARY VERDICT")

    p(f"\n  r_self discrimination  (component d between extreme modes):")
    if len(mode_names) >= 2:
        mask_first = trans['mode'] == mode_names[0]
        mask_last = trans['mode'] == mode_names[-1]
        d_self = cohens_d(trans['r_self'][mask_first], trans['r_self'][mask_last])
        d_global = cohens_d(trans['r_global'][mask_first], trans['r_global'][mask_last])
        p(f"    r_self:   d = {d_self:+.4f}  {'✓ Strong' if abs(d_self) > 0.5 else '✗ Weak'}")
        p(f"    r_global: d = {d_global:+.4f}  {'✓ Strong' if abs(d_global) > 0.5 else '✗ Weak'}")

        if abs(d_global) < 0.2:
            p(f"\n  ⚠ r_global has NEGLIGIBLE discrimination (d = {d_global:+.4f}).")
            p(f"    R_SVO at intermediate angles (30°–60°) will be dominated by r_self")
            p(f"    and will lose separation as sin(α) increases.")
            p(f"    Consider redesigning r_global (e.g., proximity-based metric).")
        elif d_self * d_global > 0:
            p(f"\n  ✓ r_self and r_global point in the SAME direction.")
            p(f"    R_SVO will be strong at all angles 0°–90°.")
            p(f"    Note: traditional SVO semantics require opposite directions.")
        else:
            p(f"\n  ⚠ r_self and r_global point in OPPOSITE directions.")
            p(f"    R_SVO will cancel at intermediate angles (~45°).")
            p(f"    Separation will be strong at 0° and 90° but weak in between.")

    p(f"\n  Viable angles for IRL mode selection (|d| > 0.8, target vs rest):")
    for mode in mode_names:
        target_mask = trans['mode'] == mode
        rest_mask = ~target_mask
        viable = []
        for angle_deg in np.arange(0, 360, 5):
            rad = np.radians(angle_deg)
            rsvo = np.cos(rad) * trans['r_self'] + np.sin(rad) * trans['r_global']
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            if abs(d) > 0.8:
                viable.append(angle_deg)

        if viable:
            ranges = []
            start = viable[0]
            prev = viable[0]
            for v in viable[1:]:
                if v - prev > 5:
                    ranges.append(f"{start:.0f}°–{prev:.0f}°" if start != prev else f"{start:.0f}°")
                    start = v
                prev = v
            ranges.append(f"{start:.0f}°–{prev:.0f}°" if start != prev else f"{start:.0f}°")
            p(f"    {mode:<25s}: {', '.join(ranges)}")
        else:
            p(f"    {mode:<25s}: NONE — mode is not separable ✗")

    if out:
        out.close()
        print(f"\nResults saved to {output_file}")

    # ---- Optional plot ----
    if do_plot:
        try:
            plot_discriminability(trans, traj_data, mode_names, pkl_path)
        except ImportError:
            print("matplotlib not available, skipping plots")


def plot_discriminability(trans, traj_data, mode_names, pkl_path):
    """Generate discriminability plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Discriminability Analysis: {os.path.basename(pkl_path)}", fontsize=13)

    angles = np.arange(0, 360, 2.5)

    # Plot 1: Target vs rest d across angles (transition level)
    ax = axes[0, 0]
    for mode in mode_names:
        target_mask = trans['mode'] == mode
        rest_mask = ~target_mask
        ds = []
        for angle_deg in angles:
            rad = np.radians(angle_deg)
            rsvo = np.cos(rad) * trans['r_self'] + np.sin(rad) * trans['r_global']
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            ds.append(d)
        ax.plot(angles, ds, label=mode, linewidth=1.5)

    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='d=0.8 threshold')
    ax.axhline(y=-0.8, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Target α (degrees)')
    ax.set_ylabel("Cohen's d (target vs rest)")
    ax.set_title('Transition-level: Target mode vs rest')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Trajectory-level same thing
    ax = axes[0, 1]
    for mode in mode_names:
        target_mask = traj_data['mode'] == mode
        rest_mask = ~target_mask
        ds = []
        for angle_deg in angles:
            rad = np.radians(angle_deg)
            rsvo = np.cos(rad) * traj_data['mean_r_self'].astype(float) + \
                   np.sin(rad) * traj_data['mean_r_global'].astype(float)
            if target_mask.sum() > 0 and rest_mask.sum() > 0:
                d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            else:
                d = 0
            ds.append(d)
        ax.plot(angles, ds, label=mode, linewidth=1.5)

    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.8, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Target α (degrees)')
    ax.set_ylabel("Cohen's d (target vs rest)")
    ax.set_title('Trajectory-level: Target mode vs rest')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: r_self distribution per mode
    ax = axes[1, 0]
    for mode in mode_names:
        mask = trans['mode'] == mode
        ax.hist(trans['r_self'][mask], bins=50, alpha=0.5, label=mode, density=True)
    ax.set_xlabel('r_self')
    ax.set_ylabel('Density')
    ax.set_title('r_self distribution by mode')
    ax.legend(fontsize=8)

    # Plot 4: r_global distribution per mode
    ax = axes[1, 1]
    for mode in mode_names:
        mask = trans['mode'] == mode
        ax.hist(trans['r_global'][mask], bins=50, alpha=0.5, label=mode, density=True)
    ax.set_xlabel('r_global')
    ax.set_ylabel('Density')
    ax.set_title('r_global distribution by mode')
    ax.legend(fontsize=8)

    plt.tight_layout()

    save_path = pkl_path.replace('.pkl', '_discriminability.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze discriminability of a demonstration dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_discriminability.py --pkl mixed_50_50_ego_alt.pkl
  python analyze_discriminability.py --pkl mega_mix_int_cumulative.pkl --plot
  python analyze_discriminability.py --pkl dataset.pkl --output report.txt --plot
        """)

    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to the demonstration .pkl file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate discriminability plots')
    parser.add_argument('--output', type=str, default=None,
                        help='Save text output to file')

    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"Error: {args.pkl} not found")
        sys.exit(1)

    analyze(args.pkl, do_plot=args.plot, output_file=args.output)