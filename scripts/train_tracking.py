"""Train / play a PPO motion-tracking policy (ReActor lower level) on a retargeted clip.

  .venv/bin/python   scripts/train_tracking.py --clip 05_02_poses --t1 4 --steps 20e6     # train
  .venv/bin/python   scripts/train_tracking.py --clip 05_02_poses --t1 4 --resume          # continue
  .venv/bin/mjpython scripts/train_tracking.py --clip 05_02_poses --t1 4 --play            # watch latest
  .venv/bin/mjpython scripts/train_tracking.py --clip 05_02_poses --t1 4 --play --best     # watch best (eval)
  add --tag NAME to keep runs apart (outputs/tracking/<clip>_<t0>-<t1>_<tag>/)

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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from tracking_env import ROOT, TrackingEnv


def latest_checkpoint(run):
    ckpts = sorted(run.glob("model_*_steps.zip"), key=lambda f: int(f.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def vecnorm_path(ckpt):  # CheckpointCallback's naming: model_vecnormalize_<n>_steps.pkl
    return str(ckpt.with_name(ckpt.stem.replace("model_", "model_vecnormalize_") + ".pkl"))


def make_env(i, env_kw):
    return functools.partial(TrackingEnv, seed=i, **env_kw)  # picklable for SubprocVecEnv


class AdaptiveLR(BaseCallback):
    """KL-adaptive learning rate (paper Tab. 2, same rule as rsl_rl), applied once per PPO update (ADR 0005)."""

    def __init__(self, lr=1e-4, desired_kl=0.01, lo=1e-5, hi=1e-3):
        super().__init__()
        self.lr, self.kl, self.lo, self.hi = lr, desired_kl, lo, hi

    def _set(self):
        lr = float(self.lr)  # plain float in the closure: the model pickles lr_schedule on save
        self.model.lr_schedule = lambda _: lr

    def _on_training_start(self):
        self._set()

    def _on_rollout_start(self):  # runs after the previous update; its approx_kl is still in the logger
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is not None:
            if kl > 2 * self.kl:
                self.lr = max(self.lo, self.lr / 1.5)
            elif kl < self.kl / 2:
                self.lr = min(self.hi, self.lr * 1.5)
        self._set()

    def _on_step(self):
        return True


class SaveBestVecNormalize(BaseCallback):
    """EvalCallback saves best_model.zip; also save the matching observation-normalization stats."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        self.model.get_vec_normalize_env().save(str(self.path))
        return True


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
    p.add_argument("--best", action="store_true", help="--play the best evaluated model instead of the latest")
    p.add_argument("--tag", default="", help="suffix for the run folder, keeps runs apart")
    args = p.parse_args()
    ref = ROOT / "outputs/retarget" / f"{args.clip}.npz"
    run = ROOT / "outputs/tracking" / (f"{args.clip}_{args.t0:g}-{args.t1:g}" + (f"_{args.tag}" if args.tag else ""))
    run.mkdir(parents=True, exist_ok=True)
    env_kw = dict(ref_path=ref, t0=args.t0, t1=args.t1, residual_force=not args.no_wrench)

    if args.play:
        import mujoco
        import mujoco.viewer
        ckpt = run / "best_model.zip" if args.best else latest_checkpoint(run)
        vn = run / "best_vecnormalize.pkl" if args.best else vecnorm_path(ckpt)
        venv = VecNormalize.load(str(vn), DummyVecEnv([make_env(0, env_kw)]))
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
        n_steps = 1024  # 8 envs x 1024 = 8k samples/update (paper: 4096 x 24 on GPU); 256 collapsed (ADR 0005)
        model = PPO(
            "MlpPolicy", venv, device="cpu",
            n_steps=n_steps, batch_size=n_steps * args.envs // 4, n_epochs=5,  # 4 mini-batches, 5 epochs (Tab. 2)
            gamma=0.97, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0025, max_grad_norm=1.0,
            learning_rate=1e-4, target_kl=0.02,  # lr is overridden by AdaptiveLR; KL early stop as a safety net
            policy_kwargs=dict(net_arch=[512, 512, 512], activation_fn=torch.nn.ELU,  # Sec. 6
                               log_std_init=-1.0),  # start with small exploration noise, else random
            # wrench/torque penalties make early episodes negative and the policy learns to fall
        )
    model.set_logger(configure(str(run), ["stdout", "csv", "tensorboard"]))
    save_every = max(1, int(1e6) // args.envs)  # ~every 1M env steps
    eval_env = VecNormalize(VecMonitor(DummyVecEnv([make_env(1000, env_kw)])), training=False, norm_reward=False)  # same wrappers as training
    callbacks = CallbackList([
        CheckpointCallback(save_every, str(run), name_prefix="model", save_vecnormalize=True),
        AdaptiveLR(),
        # deterministic, random start frames; EvalCallback copies the obs-normalization stats before evaluating
        EvalCallback(eval_env, n_eval_episodes=10, eval_freq=save_every, best_model_save_path=str(run),
                     log_path=str(run), callback_on_new_best=SaveBestVecNormalize(run / "best_vecnormalize.pkl")),
    ])
    model.learn(int(args.steps), reset_num_timesteps=not ckpt, callback=callbacks)
    model.save(run / f"model_{model.num_timesteps}_steps")
    venv.save(str(run / f"model_vecnormalize_{model.num_timesteps}_steps.pkl"))
    print("done:", run)


if __name__ == "__main__":  # guard needed: SubprocVecEnv spawns workers that re-import this file
    main()
