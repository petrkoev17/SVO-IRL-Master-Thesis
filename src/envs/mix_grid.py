import pickle


def main():
    # Load the datasets
    with open('expert_demonstrations/expert_egoistic.pkl', 'rb') as f:
        data_ego = pickle.load(f)

    with open('expert_demonstrations/expert_altruistic.pkl', 'rb') as f:
        data_pros = pickle.load(f)

    # Combine them (10 trajectories + 10 trajectories = 20 trajectories)
    mixed_data = data_ego + data_pros

    # Save the new mixed dataset
    with open('expert_data/expert_mixed_v2.pkl', 'wb') as f:
        pickle.dump(mixed_data, f)

    print(f"Created mixed dataset with {len(mixed_data)} total trajectories.")


if __name__ == "__main__":
    main()