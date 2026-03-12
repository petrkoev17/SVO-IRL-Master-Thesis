import torch
import time
import argparse
import numpy as np
from src.envs.gridworld import SVOIntersectionEnv
from src.algorithms.iq_learner import IQLearnTrainer  # Or wherever your DQNNetwork is defined
from src.algorithms.iq_learner import DQNNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the .pt model')
    args = parser.parse_args()

    # 1. Setup Environment
    env = SVOIntersectionEnv(render_mode="console")
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    # 2. Load Network
    device = 'cpu'
    network = DQNNetwork(state_dim, action_dim, hidden_dims=[64, 64]).to(device)

    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    network.load_state_dict(checkpoint['q_network'])
    network.eval()

    print(f"\nLoaded model from: {args.model_path}")
    print("Starting rollout...\n")

    # 3. Rollout Loop
    obs, _ = env.reset()
    env.render()

    done = False
    total_reward = 0

    while not done:
        time.sleep(0.5)  # Pause so you can watch it

        # Select action
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = network(obs_tensor).argmax(dim=1).item()

        action_name = "Forward" if action == 1 else "Wait"
        print(f"\nAgent chose: {action_name}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        env.render()

    print(f"\nEpisode Finished! Total Ego Reward: {total_reward}")


if __name__ == "__main__":
    main()