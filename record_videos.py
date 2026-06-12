"""Record MP4 videos: best seed per variant, 10 episodes per phase (3 phases)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from envs.parametric_lunarlander import ParametricLunarLander

PHASES = {
    1: {"gravity": -5.0,  "wind_power": 0.0,  "turbulence_power": 0.0},
    2: {"gravity": -8.0,  "wind_power": 10.0, "turbulence_power": 0.5},
    3: {"gravity": -10.0, "wind_power": 15.0, "turbulence_power": 1.5},
}
PHASE_NAMES = {1: "easy", 2: "medium", 3: "hard"}

BEST_SEEDS = {
    "baseline":     5,
    "baseline_low": 0,
    "threshold":    7,
    "pcer_flat_low": 0,
}

N_EPISODES = 10
OUT_DIR = "videos"
os.makedirs(OUT_DIR, exist_ok=True)


def record(variant: str, seed: int, phase: int, n_episodes: int, out_path: str):
    env = ParametricLunarLander(PHASES[phase], render_mode="rgb_array")
    model = PPO.load(f"checkpoints/{variant}_seed{seed}_final", env=env)

    frames = []
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        ep_frames = []
        while not done:
            frame = env.render()
            ep_frames.append(frame)
            action, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        frames.extend(ep_frames)
        total_rewards.append(total)
        print(f"  ep{ep+1}: {total:.1f}")

    env.close()
    mean_r = np.mean(total_rewards)
    print(f"  mean: {mean_r:.1f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    im = ax.imshow(frames[0])
    title = ax.set_title("", fontsize=9)

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
    ani.save(out_path, writer="ffmpeg", fps=50)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return mean_r


for variant, seed in BEST_SEEDS.items():
    print(f"\n{'='*50}")
    print(f"{variant} (seed{seed})")
    for phase in [1, 2, 3]:
        phase_name = PHASE_NAMES[phase]
        out_path = os.path.join(OUT_DIR, f"{variant}_phase{phase}_{phase_name}.mp4")
        print(f"\n  Phase {phase} ({phase_name})")
        record(variant, seed, phase, N_EPISODES, out_path)

print("\nDone. Videos saved to", OUT_DIR)
