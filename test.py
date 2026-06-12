import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import PPO
from envs.parametric_lunarlander import ParametricLunarLander


def bootstrap_ci(
    rewards: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return (lo, hi) bootstrap confidence interval for mean of rewards."""
    rng = np.random.default_rng(seed)
    arr = np.array(rewards)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def welch_ttest(rewards_a: list[float], rewards_b: list[float]) -> float:
    """Return two-sided p-value (Welch t-test) for mean(a) != mean(b)."""
    _, p = stats.ttest_ind(rewards_a, rewards_b, equal_var=False)
    return float(p)


PHASES = {
    1: {"gravity": -5.0,  "wind_power": 0.0,  "turbulence_power": 0.0},
    2: {"gravity": -8.0,  "wind_power": 10.0, "turbulence_power": 0.5},
    3: {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
}
LANDING_THRESHOLD = 200


def evaluate_model(path: str, n_episodes: int, render: bool, video_path: str | None = None, phase: int = 3) -> dict:
    env = ParametricLunarLander(PHASES[phase], render_mode="rgb_array" if (render or video_path) else None)
    model = PPO.load(path, env=env)

    rewards = []
    successes = 0
    frames = [] if video_path else None

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if frames is not None:
                frames.append(env.render())
        rewards.append(total_reward)
        if total_reward >= LANDING_THRESHOLD:
            successes += 1

    env.close()

    if video_path and frames:
        import matplotlib.animation as animation
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        im = ax.imshow(frames[0])
        def update(frame):
            im.set_data(frame)
            return [im]
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
        ani.save(video_path, writer="ffmpeg", fps=50)
        plt.close(fig)
        print(f"Video saved to {video_path}")
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "success_rate": successes / n_episodes,
        "raw_rewards": rewards,
    }


def print_results(name: str, stats: dict):
    ci_str = ""
    if "ci_lo" in stats:
        ci_str = f" [{stats['ci_lo']:.1f}, {stats['ci_hi']:.1f}]"
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Mean reward : {stats['mean']:>8.1f} ± {stats['std']:.1f}{ci_str}")
    print(f"  Min / Max   : {stats['min']:>8.1f} / {stats['max']:.1f}")
    print(f"  Success rate: {stats['success_rate']*100:>5.1f}%  (reward >= {LANDING_THRESHOLD})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true", help="Show the lander playing (deprecated, use --video)")
    parser.add_argument("--video", type=str, default=None, help="Save a video, e.g. --video baseline.mp4")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=3, help="Difficulty phase to evaluate on (1=easy, 2=medium, 3=hard)")
    parser.add_argument(
        "--variant",
        choices=["threshold", "threshold_retreat", "entropy", "td_error", "success_rate",
                 "success_rate_v1", "success_rate_v2", "continuous", "baseline", "all"],
        default="all",
    )
    parser.add_argument("--best", action="store_true", help="Load _best checkpoints instead of _final")
    args = parser.parse_args()

    variants = ["threshold", "success_rate", "continuous", "baseline"] if args.variant == "all" else [args.variant]

    p = PHASES[args.phase]
    print(f"\nEvaluating on Phase {args.phase} (gravity={p['gravity']}, wind={p['wind_power']}, turbulence={p['turbulence_power']})")
    print(f"Episodes per model: {args.episodes} | Seeds: {args.seeds}")

    results = {}
    for variant in variants:
        all_rewards = []
        all_successes = []
        raw_rewards = []
        for seed in args.seeds:
            suffix = "best" if args.best else "final"
            path = f"checkpoints/{variant}_seed{seed}_{suffix}"
            video = args.video if (args.video and seed == args.seeds[0] and variant == variants[0]) else None
            result = evaluate_model(path, args.episodes, render=args.render, video_path=video, phase=args.phase)
            all_rewards.append(result["mean"])
            all_successes.append(result["success_rate"])
            raw_rewards.extend(result.get("raw_rewards", []))

        lo, hi = bootstrap_ci(raw_rewards) if raw_rewards else (float("nan"), float("nan"))
        combined = {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "min": min(all_rewards),
            "max": max(all_rewards),
            "success_rate": float(np.mean(all_successes)),
            "ci_lo": lo,
            "ci_hi": hi,
            "raw_rewards": raw_rewards,
        }
        results[variant] = combined
        print_results(variant.upper(), combined)

    print(f"\n{'─'*40}")
    print("  Significance vs Baseline (Welch t-test)")
    print(f"{'─'*40}")
    if "baseline" in results and "raw_rewards" in results.get("baseline", {}):
        baseline_rewards = results["baseline"]["raw_rewards"]
        for variant in variants:
            if variant == "baseline":
                continue
            if "raw_rewards" not in results.get(variant, {}):
                continue
            p = welch_ttest(results[variant]["raw_rewards"], baseline_rewards)
            sig = "* (p<0.05)" if p < 0.05 else "  (n.s.)"
            print(f"  {variant:15s} p={p:.4f} {sig}")

    print(f"\n{'─'*40}\n")

    for v in results:
        results[v].pop("raw_rewards", None)
        results[v].pop("ci_lo", None)
        results[v].pop("ci_hi", None)

    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    ablation_variants = {"success_rate_v1", "success_rate_v2"}
    if set(variants) & ablation_variants:
        ablation_path = "results_ablation.json"
        existing = {}
        if os.path.exists(ablation_path):
            with open(ablation_path) as f:
                existing = json.load(f)
        existing.update({v: results[v] for v in variants if v in ablation_variants})
        with open(ablation_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Ablation results saved to {ablation_path}")
