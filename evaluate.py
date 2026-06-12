import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(y, window=50):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def load_run(log_dir: str, run_name: str):
    """Load a single Monitor CSV by name, return (timesteps_cumulative, rewards) arrays."""
    pattern = os.path.join(log_dir, f"{run_name}.monitor.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No monitor file matching {pattern}")
    df = pd.read_csv(files[0], skiprows=1)
    timesteps = df["l"].cumsum().to_numpy()
    rewards = df["r"].to_numpy()
    return timesteps, rewards


def sample_efficiency(rewards: np.ndarray, timesteps: np.ndarray, target: float = 200.0) -> int | None:
    """Return the timestep at which reward first reaches target, or None."""
    idx = np.argmax(rewards >= target)
    if rewards[idx] >= target:
        return int(timesteps[idx])
    return None


def plot_learning_curves(log_dir: str, seeds: list[int], output: str):
    variants = ["threshold", "success_rate", "continuous", "baseline"]
    colors = {
        "threshold": "steelblue",
        "success_rate": "green",
        "continuous": "crimson",
        "baseline": "gray",
    }
    window = 50

    fig, ax = plt.subplots(figsize=(10, 5))

    print("\nSample efficiency (timesteps to first reward >= 200):")
    for variant in variants:
        all_rewards = []
        all_timesteps = []
        min_len = None
        efficiencies = []
        for seed in seeds:
            try:
                x, y = load_run(log_dir, f"{variant}_seed{seed}")
                smoothed = smooth(y, window)
                all_rewards.append(smoothed)
                all_timesteps.append(x)
                min_len = len(smoothed) if min_len is None else min(min_len, len(smoothed))
                eff = sample_efficiency(y, x)
                efficiencies.append(eff if eff is not None else -1)
            except FileNotFoundError:
                print(f"Warning: missing run {variant}_seed{seed}, skipping")

        if not all_rewards:
            continue

        valid_effs = [e for e in efficiencies if e > 0]
        if valid_effs:
            print(f"  {variant}: {int(np.mean(valid_effs)):,} ± {int(np.std(valid_effs)):,} steps "
                  f"({len(valid_effs)}/{len(seeds)} seeds reached target)")
        else:
            print(f"  {variant}: never reached target reward")

        all_rewards = np.array([r[:min_len] for r in all_rewards])
        mean = all_rewards.mean(axis=0)
        std = all_rewards.std(axis=0)
        xs = np.arange(min_len)

        ax.plot(xs, mean, label=variant, color=colors[variant])
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=colors[variant])

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Mean Episode Reward (smoothed)")
    ax.set_title("PPO: Curriculum vs. Baseline (4-way comparison)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nSaved plot to {output}")


def plot_bar_chart(results: dict, output: str):
    """Bar chart of mean reward with std error bars per scheduler variant."""
    variants = [v for v in ["threshold", "success_rate", "continuous", "baseline"] if v in results]
    colors = {
        "threshold": "steelblue",
        "success_rate": "green",
        "continuous": "crimson",
        "baseline": "gray",
    }
    means = [results[v]["mean"] for v in variants]
    stds = [results[v]["std"] for v in variants]

    if not variants:
        print(f"Warning: no variants found in results, skipping bar chart")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(variants, means, yerr=stds, capsize=6,
                  color=[colors[v] for v in variants], alpha=0.85, edgecolor="black")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(200, color="black", linestyle="--", linewidth=1, label="Solved threshold (200)")
    ax.set_ylabel("Mean Episode Reward ± Std")
    ax.set_title("Final Performance Comparison: Phase 3 (Hard)")
    ax.legend()
    ax.set_ylim(0, max(means) + max(stds) + 30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart to {output}")


def plot_phase_progression(log_dir: str, seeds: list[int], output: str):
    """Plot phase index over timesteps for each curriculum scheduler."""
    import csv

    variants = ["threshold", "success_rate", "continuous"]
    colors = {"threshold": "steelblue", "success_rate": "green", "continuous": "crimson"}

    fig, ax = plt.subplots(figsize=(10, 4))
    found_any = False

    for variant in variants:
        all_events: list[tuple[int, int]] = []
        for seed in seeds:
            path = os.path.join(log_dir, f"{variant}_seed{seed}_phases.csv")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["timestep"] == "timestep":
                        continue
                    all_events.append((int(row["timestep"]), int(row["phase"])))

        if not all_events:
            print(f"Warning: no phase log found for {variant}, skipping")
            continue

        found_any = True
        all_events.sort()
        # Build step function: starts at phase 0, timestep 0
        xs = [0] + [e[0] for e in all_events]
        ys = [0] + [e[1] for e in all_events]
        ax.step(xs, ys, where="post", label=variant, color=colors[variant], linewidth=2)

    if not found_any:
        print("Warning: no phase logs found — run training with updated code to generate them")
        plt.close(fig)
        return

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Curriculum Phase")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["0 – Easy", "1 – Medium", "2 – Hard"])
    ax.set_title("Curriculum Phase Progression Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved phase progression plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="learning_curves.png")
    args = parser.parse_args()

    plot_learning_curves(args.log_dir, args.seeds, args.output)

    import json
    results_path = "results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        bar_output = args.output.replace("learning_curves", "bar_chart")
        if bar_output == args.output:
            bar_output = args.output.replace(".png", "_bar.png")
        plot_bar_chart(results, bar_output)
    else:
        print(f"Warning: {results_path} not found, skipping bar chart")

    phase_output = args.output.replace("learning_curves", "phase_progression")
    if phase_output == args.output:
        phase_output = args.output.replace(".png", "_phases.png")
    plot_phase_progression(args.log_dir, args.seeds, phase_output)
