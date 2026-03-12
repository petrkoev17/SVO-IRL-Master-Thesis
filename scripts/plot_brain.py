import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


from src.algorithms.iq_learner import DQNNetwork


def compute_preference_grid(model_path, device, grid_size=5):
    """Loads a model and calculates Q(Forward) - Q(Wait) across positions."""
    # Initialize network
    state_dim = 4
    action_dim = 2
    network = DQNNetwork(state_dim, action_dim, hidden_dims=[64, 64]).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint['q_network'])
    network.eval()

    # Grid to store preferences: [other_y, ego_x]
    preference_grid = np.zeros((grid_size, grid_size))

    with torch.no_grad():
        for ego_x in range(grid_size):
            for other_y in range(grid_size):
                # Fixed paths: Ego moves along y=2, Other moves along x=2
                state = np.array([ego_x, 2, 2, other_y], dtype=np.float32)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                q_vals = network(state_tensor).squeeze(0).cpu().numpy()
                q_wait = q_vals[0]
                q_forward = q_vals[1]

                # Preference > 0 means Forward, < 0 means Wait
                preference_grid[other_y, ego_x] = q_forward - q_wait

    return preference_grid


def main():
    parser = argparse.ArgumentParser(description="Plot Q-value heatmaps")
    parser.add_argument('--baseline', type=str, default='./runs/model_e10.pt',
                        help='Path to the unregularized model')
    parser.add_argument('--regularized', type=str, default='./runs/model_e11.pt',
                        help='Path to the SVO-regularized model')
    parser.add_argument('--save-dir', type=str, default='./runs',
                        help='Directory to save the plot')
    args = parser.parse_args()

    device = 'cpu'

    print("Evaluating Baseline Model...")
    grid_base = compute_preference_grid(args.baseline, device)

    print("Evaluating Regularized Model...")
    grid_reg = compute_preference_grid(args.regularized, device)

    # Define a shared color scale limit to make them directly comparable
    vmax = max(np.max(np.abs(grid_base)), np.max(np.abs(grid_reg)))
    vmin = -vmax

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Baseline Plot
    im0 = axes[0].imshow(grid_base, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title("Baseline IQ-Learn\nAction Preference (Forward - Wait)")
    axes[0].set_xlabel("Ego X Position")
    axes[0].set_ylabel("Other Y Position")
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))

    # Mark the conflict point
    axes[0].plot(2, 2, 'kx', markersize=15, markeredgewidth=2, label="Intersection")
    axes[0].legend(loc="upper left")

    # 2. Regularized Plot
    im1 = axes[1].imshow(grid_reg, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title("SVO-Regularized IQ-Learn\nAction Preference (Forward - Wait)")
    axes[1].set_xlabel("Ego X Position")
    axes[1].set_ylabel("Other Y Position")
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].plot(2, 2, 'kx', markersize=15, markeredgewidth=2)

    # Add Colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Preference: <0 (Wait), >0 (Forward)")

    # Save and show
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "q_value_heatmap_4.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Q-value heatmap to {save_path}")

    plt.show()


if __name__ == "__main__":
    main()