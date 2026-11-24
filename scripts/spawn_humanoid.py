"""Spawn the MuJoCo humanoid (assets/humanoid.xml).

  .venv/bin/python scripts/spawn_humanoid.py            # print skeleton, save outputs/*.png, simulate 2 s
  .venv/bin/python scripts/spawn_humanoid.py --view     # interactive viewer (plain python; mjpython is only for launch_passive)
"""
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "assets" / "humanoid.xml"
OUT = ROOT / "outputs"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Skeleton summary: this body tree is the retargeting target.
print(f"nq={model.nq} nv={model.nv} nu={model.nu} bodies={model.nbody}")
for b in range(1, model.nbody):
    parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b])
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])]
    print(f"  {model.body(b).name:16s} <- {parent:16s} z={data.xpos[b][2]:.2f}  {joints}")

if "--view" in sys.argv:
    import mujoco.viewer
    mujoco.viewer.launch(model, data)
    sys.exit()

# Headless: snapshot at spawn, then check it falls and settles without blowing up (no control).
from PIL import Image

OUT.mkdir(exist_ok=True)

cam = mujoco.MjvCamera()
cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (0, 0, 0.9), 3.5, 120, -15
with mujoco.Renderer(model, 480, 640) as r:
    r.update_scene(data, cam)
    Image.fromarray(r.render()).save(OUT / "spawn.png")
    while data.time < 2.0:
        mujoco.mj_step(model, data)
    r.update_scene(data, cam)
    Image.fromarray(r.render()).save(OUT / "after_2s.png")

head_z = data.body("head").xpos[2]
print(f"after 2 s passive: head z={head_z:.2f} (fell over, as expected with zero torque)")
assert 0.0 < head_z < 2.0, "simulation diverged"
