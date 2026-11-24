"""Play an AMASS clip as a stick-figure skeleton in the MuJoCo viewer.

  .venv/bin/mjpython scripts/play_amass.py [clip.npz]         # live viewer (passive viewer -> mjpython on macOS)
  .venv/bin/python   scripts/play_amass.py [clip.npz] --png   # headless: save outputs/<clip>.png

Uses the SMPL+H body model at dataset/smpl/{male,female}/model.npz if present
(mano.is.tue.mpg.de, "Extended SMPL+H model"), else an approximate skeleton.
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from amass import NJ, PARENTS, ROOT, load_clip

FLOOR_XML = """
<mujoco><visual><global offwidth="1280" offheight="960"/><headlight ambient=".4 .4 .4" diffuse=".6 .6 .6"/></visual>
<asset><texture name="g" type="2d" builtin="checker" width="512" height="512" rgb1=".2 .3 .4" rgb2=".1 .2 .3"/>
<material name="g" texture="g" texrepeat="8 8"/></asset>
<worldbody><light pos="0 0 4" dir="0 0 -1"/><geom type="plane" size="10 10 .1" material="g"/></worldbody>
</mujoco>"""


def draw_skeleton(scn, joints, rgba=(0.9, 0.5, 0.2, 1.0)):
    """Append one capsule per bone to an mjvScene."""
    for j in range(1, NJ):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array(rgba, dtype=float))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025, joints[PARENTS[j]], joints[j])
        scn.ngeom += 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    clip = Path(args[0]) if args else ROOT / "dataset/CMU/05/05_02_poses.npz"
    P, fps = load_clip(clip)
    print(f"{clip.name}: {len(P)} frames @ {fps:.0f} fps")

    model = mujoco.MjModel.from_xml_string(FLOOR_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # place the floor geom

    if "--png" in sys.argv:
        from PIL import Image

        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = P[0, 0], 3.5, 120, -15
        imgs = []
        with mujoco.Renderer(model, 480, 400) as r:
            for f in (0, len(P) // 3, 2 * len(P) // 3):
                r.update_scene(data, cam)
                draw_skeleton(r.scene, P[f])
                imgs.append(r.render())
        out = ROOT / "outputs" / f"{clip.stem}.png"
        out.parent.mkdir(exist_ok=True)
        Image.fromarray(np.concatenate(imgs, axis=1)).save(out)
        print("saved", out)
        sys.exit()

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:], v.cam.distance = P[0, 0], 4.0
        t0 = time.time()
        while v.is_running():
            f = int((time.time() - t0) * fps) % len(P)  # loop the clip in real time
            with v.lock():
                v.user_scn.ngeom = 0
                draw_skeleton(v.user_scn, P[f])
            v.sync()
            time.sleep(1 / 60)
