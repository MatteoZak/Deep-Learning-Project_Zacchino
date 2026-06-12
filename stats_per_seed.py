"""Per-seed statistical analysis of Phase 3 (hard) performance.

Fixes the pseudoreplication issue: the statistical unit is the SEED, not the
individual episode. For each seed we compute the mean over N_EPISODES
deterministic rollouts on Phase 3 (same protocol as run_generalization.py),
yielding one number per seed. Tests are then run on these per-seed means.
"""
import os
import json
import argparse
import numpy as np
from scipy import stats
from stable_baselines3 import PPO
from envs.parametric_lunarlander import ParametricLunarLander

PHASE3 = {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5}
N_EPISODES = 30


def seed_mean(variant: str, seed: int, n_episodes: int = N_EPISODES) -> float:
    """Mean reward over n_episodes deterministic rollouts on Phase 3 for one seed.

    Evaluation episodes use FIXED reset seeds (0..n_episodes-1) so the set of
    terrain layouts is identical across variants and reproducible across runs.
    Without this, the layout RNG adds noise that makes small effects (~5 pts)
    flip in and out of significance between runs.
    """
    env = ParametricLunarLander(PHASE3, render_mode=None)
    model = PPO.load(f"checkpoints/{variant}_seed{seed}_final", env=env)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def available_seeds(variant: str, seeds: list[int]) -> list[int]:
    return [s for s in seeds if os.path.exists(f"checkpoints/{variant}_seed{s}_final.zip")]


def per_seed_means(variant: str, seeds: list[int]) -> np.ndarray:
    out = []
    for s in available_seeds(variant, seeds):
        m = seed_mean(variant, s)
        out.append(m)
        print(f"  {variant} seed{s}: {m:.1f}")
    return np.array(out)


def bootstrap_ci_diff(a, b, n=10000, seed=42):
    """Bootstrap 95% CI for mean(a) - mean(b), resampling at the seed level."""
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a), replace=True).mean()
             - rng.choice(b, len(b), replace=True).mean()
             for _ in range(n)]
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def compare(name_a, a, name_b, b):
    # Mann-Whitney (rank-based, no normality assumption) on per-seed means
    _, p_mw = stats.mannwhitneyu(a, b, alternative="greater")
    # Welch t-test as a parametric cross-check
    _, p_t = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    lo, hi = bootstrap_ci_diff(a, b)
    diff = a.mean() - b.mean()
    sig = "***" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."
    print(f"\n=== {name_a} vs {name_b} ===")
    print(f"  {name_a:16s} mean : {a.mean():.1f}  (n={len(a)}, std={a.std(ddof=1):.1f})")
    print(f"  {name_b:16s} mean : {b.mean():.1f}  (n={len(b)}, std={b.std(ddof=1):.1f})")
    print(f"  Difference            : {diff:+.1f}")
    print(f"  95% CI (bootstrap)    : [{lo:.1f}, {hi:.1f}]")
    print(f"  Mann-Whitney U  p     : {p_mw:.4f}  {sig}")
    print(f"  Welch t-test    p     : {p_t:.4f}")
    return {
        "diff": round(diff, 2), "ci": [round(lo, 2), round(hi, 2)],
        "p_mannwhitney": round(float(p_mw), 4), "p_welch_t": round(float(p_t), 4),
        "mean_a": round(float(a.mean()), 2), "mean_b": round(float(b.mean()), 2),
        "n": len(a),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    args = ap.parse_args()
    seeds = args.seeds
    print(f"Per-seed Phase 3 evaluation (up to {len(seeds)} seeds x {N_EPISODES} "
          f"deterministic eps; each group uses all its available checkpoints)\n")

    print("[pcer_flat_low]")
    pcer = per_seed_means("pcer_flat_low", seeds)
    print("[baseline_low (oracle)]")
    base_low = per_seed_means("baseline_low", seeds)
    print("[baseline]")
    base = per_seed_means("baseline", seeds)

    res = {
        "requested_seeds": len(seeds),
        "n_episodes_per_seed": N_EPISODES,
        "per_seed_means": {
            "pcer_flat_low": [round(x, 1) for x in pcer.tolist()],
            "baseline_low": [round(x, 1) for x in base_low.tolist()],
            "baseline": [round(x, 1) for x in base.tolist()],
        },
        "pcer_flat_low_vs_baseline_low": compare("PCER-flat-low", pcer, "Baseline-low", base_low),
        "pcer_flat_low_vs_baseline": compare("PCER-flat-low", pcer, "Baseline", base),
    }

    with open("results_stats_per_seed.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved -> results_stats_per_seed.json")
