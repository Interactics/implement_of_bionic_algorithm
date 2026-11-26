"""Plot a training run's progress.csv (TensorBoard-style curves) to <run>/curves.png.

  .venv/bin/python scripts/plot_training.py outputs/tracking/05_02_poses_0-4
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [("rollout/ep_len_mean", "episode length (steps; ~half the clip = no falls, RSI)"),
          ("rollout/ep_rew_mean", "episode return"),
          ("train/value_loss", "value loss (log)"),
          ("train/clip_fraction", "PPO clip fraction"),
          ("train/approx_kl", "approx KL"),
          ("train/std", "policy std")]

run = Path(sys.argv[1])
rows = list(csv.DictReader(open(run / "progress.csv")))
x = [int(r["time/total_timesteps"]) / 1e6 for r in rows]
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
for ax, (key, title) in zip(axes.flat, PANELS):
    pts = [(xi, float(r[key])) for xi, r in zip(x, rows) if r.get(key)]
    ax.plot(*zip(*pts), lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlabel("environment steps (M)")
    ax.tick_params(labelbottom=True)
    if "log" in title:
        ax.set_yscale("log")
fig.suptitle(run.name)
fig.tight_layout()
fig.savefig(run / "curves.png", dpi=120)
print(run / "curves.png")
