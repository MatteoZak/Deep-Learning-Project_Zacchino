"""Run generalization evaluation: all schedulers × all phases × all seeds."""
import json
import numpy as np
from stable_baselines3 import PPO
from envs.parametric_lunarlander import ParametricLunarLander

PHASES = {
    1: {"gravity": -5.0,  "wind_power": 0.0,  "turbulence_power": 0.0},
    2: {"gravity": -8.0,  "wind_power": 10.0, "turbulence_power": 0.5},
    3: {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
}
SCHEDULERS = ["baseline", "threshold", "threshold_retreat", "continuous", "success_rate"]
SEEDS = list(range(10))
N_EPISODES = 30

def evaluate(variant: str, seed: int, phase: int) -> float:
    path = f"checkpoints/{variant}_seed{seed}_final"
    env = ParametricLunarLander(PHASES[phase], render_mode=None)
    model = PPO.load(path, env=env)
    rewards = []
    for _ in range(N_EPISODES):
        obs, _ = env.reset()
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


results = {}
for sched in SCHEDULERS:
    results[sched] = {}
    for phase in [1, 2, 3]:
        seed_means = []
        for seed in SEEDS:
            mean_r = evaluate(sched, seed, phase)
            seed_means.append(mean_r)
            print(f"  {sched} seed{seed} phase{phase}: {mean_r:.1f}")
        results[sched][str(phase)] = round(float(np.mean(seed_means)), 1)
        print(f"  >>> {sched} phase{phase} mean: {results[sched][str(phase)]}")

with open("results_generalization.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results_generalization.json")
print(json.dumps(results, indent=2))
