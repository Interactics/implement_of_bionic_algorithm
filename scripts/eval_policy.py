"""Evaluate a trained tracking policy against the IK-only baseline on the same segment.

  .venv/bin/python scripts/eval_policy.py outputs/tracking/07_01_poses_0-2.6_q

Run folder name gives clip and segment (<clip>_<t0>-<t1>[_tag]). Uses best_model.zip.
Robustness: deterministic rollouts from about 20 evenly spaced start frames (falls).
Quality: one rollout from frame 0 -> body error, helper force, and metrics.py (Tab. 3) for RL vs IK.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from amass import ROOT, load_clip
from metrics import evaluate, reference_contact
from tracking_env import TrackingEnv

run = Path(sys.argv[1])
clip, seg = run.name.split("_0-")[0], run.name.split("_poses_")[1].split("_")[0]
t0, t1 = map(float, seg.split("-"))
ref = ROOT / "outputs/retarget" / f"{clip}.npz"
env = TrackingEnv(str(ref), t0, t1, seed=0)
vn = VecNormalize.load(str(run / "best_vecnormalize.pkl"), DummyVecEnv([lambda: TrackingEnv(str(ref), t0, t1)]))
vn.training = False
model = PPO.load(run / "best_model.zip", device="cpu")


def roll(f0):
    env.reset(seed=f0)
    env.f = f0
    env.data.qpos[:], env.data.qvel[:] = env.ref_q[f0], env.ref_v[f0]
    mujoco.mj_forward(env.model, env.data)
    o, Q, E, F = env._obs(), [env.data.qpos.copy()], [], []
    while True:
        a, _ = model.predict(vn.normalize_obs(o[None]), deterministic=True)
        o, _, term, trunc, _ = env.step(a[0])
        Q.append(env.data.qpos.copy())
        E.append(np.linalg.norm(env.data.xpos[1:] - env.ref_x[env.f], axis=1).mean())
        F.append(np.linalg.norm(env.data.xfrc_applied[env.torso][:3]))
        if term or trunc:
            return term, np.array(Q), E, F


starts = range(0, env.T - 10, max(1, (env.T - 10) // 20))  # about 20 start frames
falls = [f0 for f0 in starts if roll(f0)[0]]
fell, Q_rl, E, F = roll(0)

# reference contact from the source clip, sampled at the policy rate (50 Hz) and at the IK rate
r = np.load(ref)
P, fps = load_clip(ROOT / str(r["source"]))
C = reference_contact(P, fps)
dt = 0.02
idx_rl = np.minimum(np.round((t0 + np.arange(len(Q_rl)) * dt) * fps).astype(int), len(P) - 1)
i0, i1 = int(t0 * fps), min(int(t1 * fps), len(P))
m_rl = evaluate(Q_rl, 1 / dt, C[idx_rl])
m_ik = evaluate(r["qpos"][i0:i1], fps, C[i0:i1])

print(f"{run.name}: falls {len(falls)}/{len(starts)} (start s: {[round(f * dt, 2) for f in falls]}); "
      f"from frame 0: {'FELL' if fell else 'reached end'}, body err {100 * np.mean(E):.1f} / {100 * np.max(E):.1f} cm, "
      f"helper force {np.mean(F):.0f} / {np.max(F):.0f} N")
print(f"{'':4s} {'ground pen time / cm':>20s} | {'self pen time / cm':>18s} | {'slide cm/s':>10s} | {'float cm':>8s}")
for name, m in (("IK", m_ik), ("RL", m_rl)):
    print(f"{name:4s} {m['ground_time']:9.2f} / {m['ground_cm']:6.1f}    | {m['self_time']:7.2f} / {m['self_cm']:6.1f}    | "
          f"{m['slide_cm_s']:10.1f} | {m['float_cm']:8.1f}")
