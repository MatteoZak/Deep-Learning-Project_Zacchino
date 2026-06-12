"""Visualize phase transition patterns across schedulers and seeds."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

SCHEDULERS = {
    "threshold": ("Threshold", "#1f77b4"),
    "threshold_retreat": ("Threshold+Retreat", "#d62728"),
}
SEEDS = list(range(10))
LOG_DIR = "runs"
PHASE_COLORS = ["#90EE90", "#FFA500", "#FF4444"]
PHASE_LABELS = ["Phase 0\n(Easy)", "Phase 1\n(Medium)", "Phase 2\n(Hard)"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for ax_idx, (sched, (label, color)) in enumerate(SCHEDULERS.items()):
    ax = axes[ax_idx]
    ax.set_title(f"{label}", fontsize=12, fontweight="bold")

    for seed in SEEDS:
        phases_path = f"{LOG_DIR}/{sched}_seed{seed}_phases.csv"
        if not os.path.exists(phases_path):
            continue

        phase_df = pd.read_csv(phases_path, header=0)
        phase_df = phase_df[pd.to_numeric(phase_df["timestep"], errors="coerce").notna()]
        phase_df["timestep"] = phase_df["timestep"].astype(int)
        phase_df["phase"] = phase_df["phase"].astype(int)
        transitions = list(zip(phase_df["timestep"], phase_df["phase"]))
        # Build segments: (start_step, end_step, phase)
        segments = []
        prev_ts = 0
        prev_phase = 0
        for ts, new_phase in transitions:
            segments.append((prev_ts, ts, prev_phase))
            prev_phase = new_phase
            prev_ts = ts
        segments.append((prev_ts, 3_000_000, prev_phase))

        y_pos = seed
        for start, end, phase in segments:
            ax.barh(y_pos, end - start, left=start, height=0.7,
                    color=PHASE_COLORS[phase], alpha=0.8, edgecolor="none")

        # Mark transition points
        for ts, new_phase in transitions:
            marker = "^" if new_phase > 0 else "v"
            ax.scatter(ts, y_pos, marker=marker, color="black", s=40, zorder=5)

    ax.set_yticks(SEEDS)
    ax.set_yticklabels([f"Seed {s}" for s in SEEDS], fontsize=8)
    ax.set_xlim(0, 3_000_000)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.5, 9.5)

axes[1].set_xlabel("Timesteps", fontsize=11)

patches = [mpatches.Patch(color=PHASE_COLORS[i], label=PHASE_LABELS[i]) for i in range(3)]
fig.legend(handles=patches, loc="upper right", fontsize=9, ncol=3,
           bbox_to_anchor=(0.98, 0.98))
fig.suptitle("Curriculum Phase Timeline per Seed\n(▲ = advance, ▼ = retreat)",
             fontsize=13, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig("phase_transitions.png", dpi=150, bbox_inches="tight")
print("Saved phase_transitions.png")
plt.close()
