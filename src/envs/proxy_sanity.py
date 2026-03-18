import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import DQN
from itertools import combinations

# --- ADJUST THESE TO YOUR SETUP ---
from src.envs.svo_proxy import SVOHighwayFlowWrapper as SVOHighwayProximityWrapper
from configs.env_config import ENV_CONFIG

EXPERTS = {
    'egoistic_0': './data/experts/expert_svo_0.0deg_seed1001/final_model.zip',
    'altruistic_90': './data/experts/expert_svo_90.0deg_seed1001/final_model.zip',
    'prosocial_45': './data/experts/expert_svo_45.0deg_seed1001/final_model.zip',
    'malicious_315': './data/experts/expert_svo_315.0deg_seed1001/final_model.zip',
}

NUM_EPISODES = 100  # increase for tighter estimates
# --- END CONFIG ---


def cohens_d(a, b):
    ps = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / (ps + 1e-8)


def collect_transitions(model_path, env_config, num_episodes):
    """Roll out an expert and collect r_self, r_global per transition."""
    env = gym.make(env_config['id'], config=env_config)
    env = SVOHighwayProximityWrapper(env, svo_alpha=0.0, lamb=1.0)

    agent = DQN.load(model_path)

    r_selfs, r_globals, speed_diffs = [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            r_selfs.append(info['rewards/component_self'])
            r_globals.append(info['rewards/component_global'])
            speed_diffs.append(info.get('rewards/speed_diff', 0.0))

    env.close()
    return {
        'r_self': np.array(r_selfs),
        'r_global': np.array(r_globals),
        'speed_diffs': np.array(speed_diffs),
    }


def main():
    # ---- Collect data from all experts ----
    results = {}
    for name, path in EXPERTS.items():
        print(f"Collecting {name}...")
        try:
            results[name] = collect_transitions(path, ENV_CONFIG, NUM_EPISODES)
            n = len(results[name]['r_self'])
            print(f"  {name}: {n} transitions")
            print(f"    r_self:   mean={results[name]['r_self'].mean():.4f}  "
                  f"std={results[name]['r_self'].std():.4f}")
            print(f"    r_global: mean={results[name]['r_global'].mean():.4f}  "
                  f"std={results[name]['r_global'].std():.4f}")
            print(f"    speed_diffs: mean={results[name]['speed_diffs'].mean():.4f}  "
                  f"std={results[name]['speed_diffs'].std():.4f}")
        except Exception as e:
            print(f"  ⚠ Failed to load {name}: {e}")

    if len(results) < 2:
        print("\nNeed at least 2 experts to compare. Exiting.")
        return

    # ---- Component-level separation (all pairs) ----
    names = list(results.keys())

    print(f"\n{'='*70}")
    print(f"PAIRWISE COHEN'S d — COMPONENTS")
    print(f"{'='*70}")

    header = f"{'Pair':<35} | {'r_self':>8} {'r_global':>9} {'speed_diffs':>9}"
    print(header)
    print("-" * len(header))

    for a, b in combinations(names, 2):
        d_self = cohens_d(results[a]['r_self'], results[b]['r_self'])
        d_global = cohens_d(results[a]['r_global'], results[b]['r_global'])
        d_dist = cohens_d(results[a]['speed_diffs'], results[b]['speed_diffs'])
        print(f"  {a} vs {b:<20s} | {d_self:+8.4f} {d_global:+9.4f} {d_dist:+9.4f}")

    # ---- R_SVO at all target angles ----
    print(f"\n{'='*70}")
    print(f"R_SVO SEPARATION (target mode vs rest of dataset)")
    print(f"{'='*70}")
    print(f"d > 0 means target mode scores higher than the rest")
    print(f"✓ = |d| > 0.8 (large effect)\n")

    # Pool all transitions into "rest" for each target
    all_r_self = np.concatenate([results[n]['r_self'] for n in names])
    all_r_global = np.concatenate([results[n]['r_global'] for n in names])
    all_modes = np.concatenate([np.full(len(results[n]['r_self']), n) for n in names])

    # Map each expert to its "natural" SVO angle
    expert_angles = {
        'egoistic_0': 0,
        'prosocial_45': 45,
        'altruistic_90': 90,
        'malicious_315': 315,
    }

    test_angles = [0, 22.5, 45, 67.5, 90, 135, 180, 225, 270, 315, 337.5]

    # Header
    angle_labels = [f"{a:>6.1f}°" for a in test_angles]
    print(f"  {'Target':<20s} | " + " ".join(angle_labels))
    print(f"  {'-'*20}-+-" + "-".join(["-" * 7] * len(test_angles)))

    for target_name in names:
        if target_name not in expert_angles:
            continue

        target_mask = all_modes == target_name
        rest_mask = ~target_mask

        ds = []
        for angle_deg in test_angles:
            a = np.radians(angle_deg)
            c, s = np.cos(a), np.sin(a)
            rsvo = c * all_r_self + s * all_r_global
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            ds.append(d)

        row = " ".join(f"{d:+7.3f}" for d in ds)
        print(f"  {target_name:<20s} | {row}")

    # ---- Best angle for each target ----
    print(f"\n{'='*70}")
    print(f"BEST ANGLE FOR EACH TARGET MODE")
    print(f"{'='*70}")

    fine_angles = np.arange(0, 360, 2.5)

    for target_name in names:
        if target_name not in expert_angles:
            continue

        target_mask = all_modes == target_name
        rest_mask = ~target_mask
        natural = expert_angles[target_name]

        best_d = -float('inf')
        best_angle = 0

        for angle_deg in fine_angles:
            a = np.radians(angle_deg)
            rsvo = np.cos(a) * all_r_self + np.sin(a) * all_r_global
            d = cohens_d(rsvo[target_mask], rsvo[rest_mask])
            if d > best_d:
                best_d = d
                best_angle = angle_deg

        # Also check d at the natural angle
        a_nat = np.radians(natural)
        rsvo_nat = np.cos(a_nat) * all_r_self + np.sin(a_nat) * all_r_global
        d_nat = cohens_d(rsvo_nat[target_mask], rsvo_nat[rest_mask])

        print(f"  {target_name}:")
        print(f"    Natural angle ({natural:5.1f}°): d = {d_nat:+.4f} {'✓' if abs(d_nat) > 0.8 else '✗'}")
        print(f"    Best angle    ({best_angle:5.1f}°): d = {best_d:+.4f} {'✓' if abs(best_d) > 0.8 else '✗'}")

    # ---- Sign check ----
    print(f"\n{'='*70}")
    print(f"SIGN CHECK — r_global means")
    print(f"{'='*70}")
    for name in names:
        print(f"  {name:<20s}: r_self={results[name]['r_self'].mean():.4f}  "
              f"r_global={results[name]['r_global'].mean():.4f}  "
              f"speed_diffs={results[name]['speed_diffs'].mean():.4f}")

    print(f"\n  Expected ordering (non-inverted, high r_global = more space):")
    print(f"    r_self:   egoistic > prosocial > altruistic")
    print(f"    r_global: egoistic > prosocial > altruistic > malicious")
    print(f"    (malicious crowds others → lowest min-distance → lowest r_global)")


if __name__ == "__main__":
    main()