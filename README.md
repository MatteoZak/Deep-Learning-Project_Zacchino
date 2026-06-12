# Entropy as a Curriculum Gate

Investigating the interaction between **entropy regularization** and **curriculum
learning** in parametric reinforcement learning, on a difficulty-parameterized
`LunarLander-v3`. Course project for *Deep Learning & Applied AI* (DLAI),
Sapienza University of Rome, a.y. 2025/26.

## TL;DR

We train PPO agents with a curriculum over three difficulty phases (increasing
gravity, wind, and turbulence) and study how the PPO entropy coefficient
interacts with threshold-based phase advancement.

- **Entropy gate.** The entropy coefficient decides whether agents ever reach the
  hardest phase: with `α = 0.10`, **0/10** seeds advance to the hard phase; with
  `α = 0.02`, **10/10** do.
- **Entropy dominates.** Lowering the entropy coefficient gives a large, robust
  gain (**+53 reward, p < 0.001**). Adding a curriculum at fixed low entropy is
  *statistically indistinguishable* from training directly on the hard
  environment (**+6.4, p = 0.054, n.s.**).

## Repository layout

```
envs/                  ParametricLunarLander (runtime-configurable difficulty)
schedulers/            6 curriculum schedulers (threshold, retreat, entropy,
                       success-rate, continuous, PCER) + base class
callbacks/             SB3 callbacks: curriculum advancement + best-model saving
configs/default.yaml   phases, scheduler params, PPO hyperparameters
train.py               train one (scheduler, seed) run
run_generalization.py  evaluate trained models zero-shot across all phases
stats_per_seed.py      per-seed statistical analysis of Phase 3 (the paper's stats)
evaluate.py            learning curves, bar chart, phase-progression plots
plot_*.py              scripts that generate the result figures
results_notebook.ipynb end-to-end results notebook (tables, heatmap, figures, stats)
tests/                 pytest unit tests (schedulers, callback, env, stats)
checkpoints/           trained PPO models (*_final.zip, *_best.zip)
```

The trained `checkpoints/` are included, so the results can be reproduced
without retraining.

## Setup

```bash
git clone https://github.com/MatteoZak/Deep-Learning-Project_Zacchino.git
cd Deep-Learning-Project_Zacchino
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14, PyTorch 2.12, Stable-Baselines3 2.8, Gymnasium 1.2.
`gymnasium[box2d]` pulls in the Box2D physics backend (via `swig` / `box2d-py`)
required by LunarLander.

## Reproducing the results

The repository ships with the trained models, so you can reproduce the analysis
directly. To recompute from the checkpoints:

```bash
# Key statistical result (per-seed Phase 3 analysis, the paper's main claim).
# Each seed -> mean over 30 deterministic episodes with fixed eval layouts.
python stats_per_seed.py            # 20 seeds for the key oracle comparison

# Zero-shot generalization across all phases (regenerates results_generalization.json)
python run_generalization.py

# Figures and tables (also reproduced in results_notebook.ipynb)
jupyter notebook results_notebook.ipynb
```

> **Statistical note.** Significance tests use the **seed** as the statistical
> unit (one per-seed mean over 30 deterministic episodes), *not* the individual
> episode. Pooling episodes would be pseudoreplication and would overstate
> significance. Evaluation episodes use fixed reset seeds so terrain layouts are
> identical across variants and results are reproducible across runs.

## Training from scratch (optional)

```bash
# Train one run: pick a scheduler and a seed
python train.py --scheduler pcer_flat_low --seed 0
python train.py --scheduler pcer          --seed 0   # high-entropy gate demo

# Available schedulers (see train.py --help):
#   threshold, threshold_retreat, entropy, td_error, success_rate,
#   continuous, pcer, pcer_high, pcer_low, pcer_flat, pcer_flat_low,
#   threshold_3phase, threshold_retreat_3phase, pcer_3phase, pcer_flat_low_3phase
```

Each run is 3M timesteps and writes a checkpoint to `checkpoints/`. Phases,
thresholds, and PPO hyperparameters live in
[`configs/default.yaml`](configs/default.yaml).

## Tests

```bash
pytest -q        # 71 unit tests: schedulers, curriculum callback, env, stats
```
