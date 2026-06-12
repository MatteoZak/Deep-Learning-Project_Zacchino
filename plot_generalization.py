"""Plot generalization heatmap from results_generalization.json."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

with open("results_generalization.json") as f:
    data = json.load(f)

LABELS = {
    "baseline": "Baseline",
    "threshold": "Threshold",
    "threshold_retreat": "Threshold+Retreat",
    "continuous": "Continuous",
    "success_rate": "SuccessRate",
    "pcer": "PCER",
    "pcer_low": "PCER-low",
    "baseline_low": "Baseline-low",
    "pcer_flat_low": "PCER-flat-low",
    "threshold_3phase": "Threshold-3ph",
    "threshold_retreat_3phase": "Threshold+Retreat-3ph",
    "pcer_3phase": "PCER-3ph",
    "pcer_flat_low_3phase": "PCER-flat-low-3ph",
}
schedulers = list(data.keys())
phases = [1, 2, 3]
phase_labels = ["Easy", "Medium", "Hard"]

matrix = np.array([[data[s][str(p)] for p in phases] for s in schedulers])

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=280)

ax.set_xticks(range(len(phases)))
ax.set_xticklabels(phase_labels, fontsize=11, rotation=45, ha="left")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.set_yticks(range(len(schedulers)))
ax.set_yticklabels([LABELS[s] for s in schedulers], fontsize=11)

for i in range(len(schedulers)):
    for j in range(len(phases)):
        val = matrix[i, j]
        color = "black" if val > 150 else "white"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=12,
                fontweight="bold", color=color)

plt.colorbar(im, ax=ax, label="Mean Reward (30 eps × 10 seeds)")
ax.set_title("Generalization: Scheduler × Difficulty Phase", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("generalization_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved generalization_heatmap.png")
plt.close()
