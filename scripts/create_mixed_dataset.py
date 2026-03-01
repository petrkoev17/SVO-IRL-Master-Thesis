from scripts.extract_demonstrations import combine_demonstrations

demo_paths = [
    'expert_demonstrations/altruistic_90_demonstrations.pkl',  # Altruistic
    'expert_demonstrations/iq_self_0_v1_demonstrations.pkl',   # Egoistic
]

combined_trajs, combined_stats = combine_demonstrations(
    demo_paths=demo_paths,
    output_path='./expert_demonstrations/mixed_50_50_ego_alt.pkl',
    weights=[0.5, 0.5]  # Equal split
)

print("\n" + "="*60)
print("Mixed Dataset Created")
print("="*60)
print(f"Total episodes: {combined_stats['num_episodes']}")
print(f"Total transitions: {combined_stats['total_transitions']}")
print(f"Mean reward: {combined_stats['mean_reward']:.3f}")
print(f"Collision rate: {combined_stats['collision_rate']:.2%}")