import pickle

files = {
    'egoistic':  'expert_demonstrations_v2/expert_egoistic.pkl',
    'prosocial': 'expert_demonstrations_v2/expert_prosocial.pkl',
    'altruistic': 'expert_demonstrations_v2/expert_altruistic.pkl',
}

mixed = []
traj_labels = []
for name, f in files.items():
    with open(f, 'rb') as fh:
        trajs = pickle.load(fh)
        n = sum(len(t) for t in trajs)
        print(f"{name}: {len(trajs)} trajectories, {n} transitions")
        mixed.extend(trajs)
        traj_labels.extend([name] * len(trajs))

total = sum(len(t) for t in mixed)
out = 'expert_demonstrations_v2/expert_mixed_all_v2.pkl'

data = {
    'trajectories': mixed,
    'metadata': {
        'source_paths': list(files.values()),
        'weights': [1/3, 1/3, 1/3],
        'mode_labels': traj_labels,
    },
}

with open(out, 'wb') as fh:
    pickle.dump(data, fh)
print(f"\nSaved mixed: {len(mixed)} trajectories, {total} transitions -> {out}")