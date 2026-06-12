"""Figure: entropy gate effect — phase transitions for pcer_original vs pcer_flat_low."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)

colors = {"pcer": "#d62728", "pcer_flat_low": "#1f77b4"}
TOTAL = 3_000_000

def load_phases(scheduler, seed):
    path = f"runs/{scheduler}_seed{seed}_phases.csv"
    try:
        df = pd.read_csv(path)
        return list(zip(df["timestep"], df["phase"]))
    except Exception:
        return []

def phase_steps(transitions, total):
    """Return (timesteps, phases) as step function."""
    xs = [0]
    ys = [0]
    for t, p in transitions:
        xs.append(t)
        ys.append(p - 1)   # 0-indexed (phase label 1 → index 0)
        xs.append(t)
        ys.append(p)
    xs.append(total)
    ys.append(ys[-1])
    return xs, ys

for ax, sched, label, color in [
    (axes[0], "pcer", "PCER (ent=0.10)", colors["pcer"]),
    (axes[1], "pcer_flat_low", "PCER-flat-low (ent=0.02)", colors["pcer_flat_low"]),
]:
    for seed in range(10):
        transitions = load_phases(sched, seed)
        xs, ys = phase_steps(transitions, TOTAL)
        ax.step(xs, ys, where="post", color=color, alpha=0.4, linewidth=1.0)

    ax.set_xlim(0, TOTAL)
    ax.set_ylim(-0.1, 2.3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Phase 1\n(easy)", "Phase 2\n(medium)", "Phase 3\n(hard)"])
    ax.set_xlabel("Training steps", fontsize=9)
    ax.set_title(label, fontsize=9, fontweight="bold", color=color)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

axes[0].set_ylabel("Curriculum phase", fontsize=9)
fig.suptitle("Entropy Gate Effect: Curriculum Progression Over Training (10 seeds each)",
             fontsize=9, y=1.02)
plt.tight_layout()
plt.savefig("phase_gate.png", dpi=150, bbox_inches="tight")
print("Saved phase_gate.png")
plt.close()
