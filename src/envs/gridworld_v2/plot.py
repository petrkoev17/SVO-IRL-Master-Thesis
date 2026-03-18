"""
Visualization for Gridworld V2 Results

Plot 1: Action preference heatmaps — for each model, show which action
        the agent picks at each (ego_x, other1_dist_to_conflict) state.

Plot 2: SVO ring — plot each model's (Σr_self, Σr_global) on the reward
        plane with SVO angle reference lines.

Usage:
    python plot_v2_results.py \
        --baseline runs/baseline.pt \
        --egoistic runs/ego_reg.pt \
        --prosocial runs/pro_reg.pt \
        --altruistic runs/alt_reg.pt \
        --save-dir ./plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.iq_learner import DQNNetwork
from src.envs.gridworld_v2.gridworld_v2 import (
    SVOIntersectionGridV2, GRID_SIZE, CONFLICT_X, CONFLICT_Y,
    SPEED_STOP, SPEED_SLOW, SPEED_FAST, NOT_SPAWNED, REACTION_DISTANCE
)


# ── Plot 1: Action Preference Heatmaps (rollout-based) ───────────────

MAX_DIST = 4  # Show distances 0..3 (reaction zone)

def collect_rollout_actions(model_path, env, num_episodes=50, device='cpu'):
    """
    Roll out the policy and record (ego_x, min_other_dist, action) at each step.
    Uses the minimum distance across all active others (closest threat).
    Returns counts[dist_bin][ego_x][action].
    dist_bin: 0=at conflict, 1=1 cell, 2=2 cells, 3=3+ cells or no other present.
    """
    state_dim = 6
    action_dim = 3
    network = DQNNetwork(state_dim, action_dim, hidden_dims=[128, 128]).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint['q_network'])
    network.eval()

    counts = np.zeros((MAX_DIST, GRID_SIZE, 3))

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + 200)
        done = False

        while not done:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = network(state_tensor)
            action = q_vals.argmax(dim=1).item()

            ego_x = int(obs[0])

            # Find minimum distance to conflict across all active others
            min_dist = MAX_DIST - 1  # default: "far / no threat"
            has_other = False
            for other_idx in [2, 4]:
                o_y = int(obs[other_idx])
                if o_y == NOT_SPAWNED:
                    continue
                has_other = True
                dist = CONFLICT_Y - o_y
                if dist >= 0:
                    min_dist = min(min_dist, dist)

            dist_bin = min(min_dist, MAX_DIST - 1)
            if dist_bin >= 0:
                counts[dist_bin, ego_x, action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return counts


def _draw_split_cell(ax, cx, cy, counts_vec, action_colors, unvisited_color, size=1.0):
    """
    Draw a single cell at (cx, cy) with diagonal splits proportional to action mix.
    counts_vec: array of shape (3,) with counts for [STOP, SLOW, FAST].
    """
    total = counts_vec.sum()
    if total == 0:
        # Unvisited — gray square
        rect = plt.Rectangle((cx - size/2, cy - size/2), size, size,
                              facecolor=unvisited_color, edgecolor='#cccccc',
                              linewidth=0.3)
        ax.add_patch(rect)
        return

    # Get actions with nonzero counts, sorted by count (largest first)
    fracs = counts_vec / total
    active = [(i, fracs[i]) for i in range(3) if fracs[i] > 0]
    active.sort(key=lambda x: -x[1])

    half = size / 2
    x0, y0 = cx - half, cy - half  # bottom-left corner

    if len(active) == 1:
        # Solid color
        act, _ = active[0]
        rect = plt.Rectangle((x0, y0), size, size,
                              facecolor=action_colors[act],
                              edgecolor='#666666', linewidth=0.4)
        ax.add_patch(rect)
    elif len(active) == 2:
        # Diagonal split: top-left triangle = first action, bottom-right = second
        act1, frac1 = active[0]
        act2, frac2 = active[1]

        if frac1 >= 0.5:
            # Dominant action gets the larger triangle (top-left to bottom-right diagonal)
            tri1 = plt.Polygon([(x0, y0), (x0, y0+size), (x0+size, y0+size)],
                               facecolor=action_colors[act1], edgecolor='none')
            tri2 = plt.Polygon([(x0, y0), (x0+size, y0+size), (x0+size, y0)],
                               facecolor=action_colors[act2], edgecolor='none')
        else:
            tri1 = plt.Polygon([(x0, y0), (x0, y0+size), (x0+size, y0+size)],
                               facecolor=action_colors[act1], edgecolor='none')
            tri2 = plt.Polygon([(x0, y0), (x0+size, y0+size), (x0+size, y0)],
                               facecolor=action_colors[act2], edgecolor='none')

        ax.add_patch(tri1)
        ax.add_patch(tri2)
        # Diagonal line
        ax.plot([x0, x0+size], [y0, y0+size], color='white', linewidth=0.6, zorder=3)
        # Border
        rect = plt.Rectangle((x0, y0), size, size, facecolor='none',
                              edgecolor='#666666', linewidth=0.4)
        ax.add_patch(rect)
    else:
        # Three actions: top-left triangle, bottom-right triangle, thin middle stripe
        act1, _ = active[0]
        act2, _ = active[1]
        act3, _ = active[2]
        tri1 = plt.Polygon([(x0, y0+size*0.3), (x0, y0+size), (x0+size, y0+size)],
                           facecolor=action_colors[act1], edgecolor='none')
        tri2 = plt.Polygon([(x0, y0), (x0+size, y0), (x0+size, y0+size*0.7)],
                           facecolor=action_colors[act2], edgecolor='none')
        tri3 = plt.Polygon([(x0, y0+size*0.3), (x0+size, y0+size*0.7),
                             (x0+size, y0), (x0, y0)],
                           facecolor=action_colors[act3], edgecolor='none')
        ax.add_patch(tri3)
        ax.add_patch(tri1)
        ax.add_patch(tri2)
        rect = plt.Rectangle((x0, y0), size, size, facecolor='none',
                              edgecolor='#666666', linewidth=0.4)
        ax.add_patch(rect)


def plot_action_heatmaps(model_paths, model_names, save_dir):
    """Thesis-quality action heatmaps with diagonal-split cells."""
    env = SVOIntersectionGridV2(deterministic_traffic=False)
    n = len(model_paths)

    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.0))
    if n == 1:
        axes = [axes]

    action_colors = {
        0: '#cc3333',   # STOP  — red
        1: '#ddaa22',   # SLOW  — amber
        2: '#339933',   # FAST  — green
    }
    unvisited_color = '#e8e8e4'

    dist_labels = ['At conflict', '1 cell away', '2 cells away', '≥3 cells']
    ego_labels = ['0\nstart', '1', '2\nyield', '3\nconflict', '4', '5', '6\ngoal']

    behavioral_labels = [
        'Always yields',
        'Charges through',
        'Yields selectively',
        'Never crosses',
    ]

    for idx, (ax, path, name) in enumerate(zip(axes, model_paths, model_names)):
        counts = collect_rollout_actions(path, env)

        # Draw each cell with diagonal splits
        for d in range(MAX_DIST):
            for ex in range(GRID_SIZE):
                _draw_split_cell(ax, ex, d, counts[d, ex],
                                 action_colors, unvisited_color)

        # Percentage annotation for visited cells
        for d in range(MAX_DIST):
            for ex in range(GRID_SIZE):
                total = int(counts[d, ex].sum())
                if total == 0:
                    continue

                fracs = counts[d, ex] / total
                active = [(i, fracs[i]) for i in range(3) if fracs[i] > 0.01]

                if len(active) == 1:
                    act, _ = active[0]
                    letter = ['S', 'W', 'F'][act]
                    ax.text(ex, d, f'{letter}\n({total})', ha='center', va='center',
                            fontsize=6.5, fontweight='bold', color='white',
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                         foreground='#333333')])
                else:
                    # Show percentages for mixed cells
                    lines = []
                    for act, frac in sorted(active, key=lambda x: -x[1]):
                        letter = ['S', 'W', 'F'][act]
                        lines.append(f'{letter}:{frac:.0%}')
                    ax.text(ex, d, '\n'.join(lines), ha='center', va='center',
                            fontsize=5.5, fontweight='bold', color='white',
                            path_effects=[pe.withStroke(linewidth=1.3,
                                                         foreground='#333333')])

        # Conflict point marker
        ax.plot(CONFLICT_X, 0, 'x', color='black', markersize=10,
                markeredgewidth=2.5, zorder=10)

        # Yield decision column highlight
        rect = plt.Rectangle((CONFLICT_X - 1 - 0.5, -0.5), 1, MAX_DIST,
                              linewidth=2, edgecolor='#FFD700', facecolor='none',
                              linestyle=(0, (4, 2)), zorder=8)
        ax.add_patch(rect)

        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, MAX_DIST - 0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_xticklabels(ego_labels, fontsize=6.5)
        ax.set_xlabel('Ego position', fontsize=9)

        ax.set_yticks(range(MAX_DIST))
        ax.set_yticklabels(dist_labels if idx == 0 else
                           ['0', '1', '2', '≥3'], fontsize=7)
        if idx == 0:
            ax.set_ylabel('Nearest other vehicle\ndistance to conflict', fontsize=9)

        ax.set_title(name, fontsize=10, fontweight='bold', pad=6)
        ax.set_aspect('equal')

        # Behavioral label inside panel (top-right corner)
        ax.text(0.97, 0.03, behavioral_labels[idx],
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, fontstyle='italic', color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.85))

    # Legend
    patches = [
        mpatches.Patch(color=action_colors[0], label='STOP'),
        mpatches.Patch(color=action_colors[1], label='SLOW'),
        mpatches.Patch(color=action_colors[2], label='FAST'),
        mpatches.Patch(color=unvisited_color, edgecolor='#aaaaaa',
                       linewidth=0.5, label='Unvisited'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True,
               fancybox=True, edgecolor='#cccccc')

    fig.suptitle('Ego yielding behavior at the intersection decision point',
                 fontsize=13, fontweight='bold', y=1.01)
    # fig.text(0.5, 0.8,
    #          'Action mix from 50 stochastic rollouts · '
    #          'Gold box = yield decision point · '
    #          '✕ = conflict · Diagonal = mixed actions',
    #          ha='center', fontsize=7.5, color='#666666',
    #          transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    save_path = os.path.join(save_dir, 'action_heatmaps.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved action heatmaps to {save_path}")
    plt.close()


# ── Plot 2: SVO Ring ─────────────────────────────────────────────────

def evaluate_model(model_path, env, num_episodes=20, device='cpu'):
    """Roll out a model and collect Σr_self, Σr_global per episode."""
    state_dim = 6
    action_dim = 3
    network = DQNNetwork(state_dim, action_dim, hidden_dims=[128, 128]).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint['q_network'])
    network.eval()

    results = {'r_self': [], 'r_global': [], 'length': [], 'arrived': []}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + 100)
        done = False
        ep_r_self = 0
        ep_r_global = 0
        steps = 0

        while not done:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = network(state_tensor)
            action = q_vals.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r_self += info['rewards/component_self']
            ep_r_global += info['rewards/component_global']
            steps += 1

        results['r_self'].append(ep_r_self)
        results['r_global'].append(ep_r_global)
        results['length'].append(steps)
        results['arrived'].append(info.get('ego_arrived', False))

    return results


def plot_svo_ring(model_paths, model_names, save_dir):
    """Plot each model's reward signature on the SVO plane."""
    env = SVOIntersectionGridV2(deterministic_traffic=False)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw SVO reference lines
    for angle_deg, label, ls in [
        (0, 'Egoistic (0°)', ':'),
        (45, 'Prosocial (45°)', '--'),
        (90, 'Altruistic (90°)', ':'),
    ]:
        angle_rad = np.deg2rad(angle_deg)
        r = 12
        ax.plot([0, r * np.cos(angle_rad)], [0, r * np.sin(angle_rad)],
                color='gray', linestyle=ls, alpha=0.4, linewidth=1)
        ax.annotate(label,
                    xy=(r * 0.85 * np.cos(angle_rad), r * 0.85 * np.sin(angle_rad)),
                    fontsize=8, color='gray', ha='center', va='center',
                    rotation=angle_deg)

    # Colors for each model
    colors = ['#555555', '#d94040', '#4080d0', '#40a040']
    markers = ['s', '^', 'o', 'D']

    for path, name, color, marker in zip(model_paths, model_names, colors, markers):
        results = evaluate_model(path, env)

        rs = np.array(results['r_self'])
        rg = np.array(results['r_global'])

        # Plot individual episodes as small dots
        ax.scatter(rs, rg, c=color, alpha=0.25, s=20, edgecolors='none')

        # Plot mean as large marker
        mean_rs = rs.mean()
        mean_rg = rg.mean()
        ax.scatter(mean_rs, mean_rg, c=color, marker=marker, s=150,
                   edgecolors='black', linewidth=1.2, label=name, zorder=5)

        # Annotate recovered angle
        recovered = np.rad2deg(np.arctan2(mean_rg, mean_rs))
        ax.annotate(f'{recovered:.0f}°', (mean_rs, mean_rg),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9, color=color, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Σ r_self (ego reward)', fontsize=12)
    ax.set_ylabel('Σ r_global (social reward)', fontsize=12)
    ax.set_title('SVO Reward Signatures — Trained Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    save_path = os.path.join(save_dir, 'svo_ring.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved SVO ring to {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot Gridworld V2 results')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline (no SVO reg) model .pt')
    parser.add_argument('--egoistic', type=str, required=True,
                        help='Path to egoistic-regularized model .pt')
    parser.add_argument('--prosocial', type=str, required=True,
                        help='Path to prosocial-regularized model .pt')
    parser.add_argument('--altruistic', type=str, required=True,
                        help='Path to altruistic-regularized model .pt')
    parser.add_argument('--save-dir', type=str, default='./plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    paths = [args.baseline, args.egoistic, args.prosocial, args.altruistic]
    names = ['Baseline\n(no reg)', 'SVO-Reg\n(egoistic)', 'SVO-Reg\n(prosocial)', 'SVO-Reg\n(altruistic)']

    print("Generating action heatmaps...")
    plot_action_heatmaps(paths, names, args.save_dir)

    print("Generating SVO ring plot...")
    plot_svo_ring(paths, names, args.save_dir)

    print("Done!")


if __name__ == "__main__":
    main()