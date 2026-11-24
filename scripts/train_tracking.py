"""Train / play a PPO motion-tracking policy (ReActor lower level) on a retargeted clip.

  .venv/bin/python   scripts/train_tracking.py --clip 05_02_poses --t1 4 --steps 20e6     # train
  .venv/bin/python   scripts/train_tracking.py --clip 05_02_poses --t1 4 --resume          # continue
  .venv/bin/mjpython scripts/train_tracking.py --clip 05_02_poses --t1 4 --play            # watch latest

Reference: outputs/retarget/<clip>.npz (run scripts/retarget.py first).
Runs land in outputs/tracking/<clip>_<t0>-<t1>/ (checkpoints, VecNormalize stats, progress.csv).
"""
import argparse
import functools
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from tracking_env import ROOT, TrackingEnv


def latest_checkpoint(run):
    ckpts = sorted(run.glob("model_*_steps.zip"), key=lambda f: int(f.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def vecnorm_path(ckpt):  # CheckpointCallback's naming: model_vecnormalize_<n>_steps.pkl
    return str(ckpt.with_name(ckpt.stem.replace("model_", "model_vecnormalize_") + ".pkl"))


def make_env(i, env_kw):
    return functools.partial(TrackingEnv, seed=i, **env_kw)  # picklable for SubprocVecEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="05_02_poses")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=4.0)
    p.add_argument("--steps", type=float, default=20e6)
    p.add_argument("--envs", type=int, default=os.cpu_count())
    p.add_argument("--no-wrench", action="store_true", help="disable the residual root wrench")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--play", action="store_true")
    args = p.parse_args()
    ref = ROOT / "outputs/retarget" / f"{args.clip}.npz"
    run = ROOT / "outputs/tracking" / f"{args.clip}_{args.t0:g}-{args.t1:g}"
    run.mkdir(parents=True, exist_ok=True)
    env_kw = dict(ref_path=ref, t0=args.t0, t1=args.t1, residual_force=not args.no_wrench)

    if args.play:
        import mujoco
        import mujoco.viewer
        from stable_baselines3.common.vec_env import DummyVecEnv

        ckpt = latest_checkpoint(run)
        venv = VecNormalize.load(vecnorm_path(ckpt), DummyVecEnv([make_env(0, env_kw)]))
        venv.training, venv.norm_reward = False, False
        model = PPO.load(ckpt, device="cpu")
        env = venv.envs[0]
        print("playing", ckpt.name, "(reference bodies = small orange spheres)")
        obs = venv.reset()
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            steps = 0
            while v.is_running():
                t = time.time()
                act, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = venv.step(act)  # auto-resets at clip end or fall
                steps = 0 if done[0] else steps + 1
                with v.lock():
                    v.user_scn.ngeom = 0
                    for x in env.ref_x[env.f]:
                        mujoco.mjv_initGeom(v.user_scn.geoms[v.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([.03, 0, 0]), x, np.eye(3).flatten(), np.array([1, .5, 0, .8]))
                        v.user_scn.ngeom += 1
                v.sync()
                time.sleep(max(0, 0.02 - (time.time() - t)))  # real time at 50 Hz
        return

    venv = VecMonitor(SubprocVecEnv([make_env(i, env_kw) for i in range(args.envs)]))
    ckpt = latest_checkpoint(run) if args.resume else None
    if ckpt:
        venv = VecNormalize.load(vecnorm_path(ckpt), venv)
        model = PPO.load(ckpt, env=venv, device="cpu")
        print("resuming from", ckpt.name)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.97)
        n_steps = 256  # ponytail: paper uses 4096 envs x 24 steps on GPU; few CPU envs need longer rollouts
        model = PPO(
            "MlpPolicy", venv, device="cpu",
            n_steps=n_steps, batch_size=n_steps * args.envs // 4, n_epochs=5,  # 4 mini-batches, 5 epochs (Tab. 2)
            gamma=0.97, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0025, max_grad_norm=1.0,
            learning_rate=3e-4, target_kl=0.02,  # ponytail: fixed lr + KL early stop, not the adaptive-lr schedule
            policy_kwargs=dict(net_arch=[512, 512, 512], activation_fn=torch.nn.ELU,  # Sec. 6
                               log_std_init=-1.0),  # start with small exploration noise, else random
            # wrench/torque penalties make early episodes negative and the policy learns to fall
        )
    model.set_logger(configure(str(run), ["stdout", "csv"]))
    save_every = max(1, int(1e6) // args.envs)  # ~every 1M env steps
    model.learn(int(args.steps), reset_num_timesteps=not ckpt,
                callback=CheckpointCallback(save_every, str(run), name_prefix="model", save_vecnormalize=True))
    model.save(run / f"model_{model.num_timesteps}_steps")
    venv.save(str(run / f"model_vecnormalize_{model.num_timesteps}_steps.pkl"))
    print("done:", run)


if __name__ == "__main__":  # guard needed: SubprocVecEnv spawns workers that re-import this file
    main()
