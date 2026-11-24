"""Pull a few clips out of the AMASS CMU archive without extracting the whole thing.

  .venv/bin/python scripts/extract_amass_clips.py dataset/CMU.tar.bz2
  -> dataset/CMU/<subject>/<clip>.npz   (then you can delete the archive)
"""
import sys
import tarfile
from pathlib import Path

# subject_trial ids (preview at mocap.cs.cmu.edu)
CLIPS = {
    "02_01": "walk",
    "07_01": "walk",
    "05_02": "modern dance: expressive arms, pirouette",
    "05_06": "modern dance: cartwheel-like start, pirouettes, jete",
    "60_01": "salsa",
}

archive = Path(sys.argv[1])
out = archive.parent  # dataset/CMU.tar.bz2 -> dataset/CMU/<subject>/<clip>.npz
found = set()
# ponytail: streams the whole archive once (bz2 isn't seekable), fine for a one-off
with tarfile.open(archive) as tar:
    for m in tar:
        name = Path(m.name).name  # e.g. 05_02_poses.npz or 05_02_stageii.npz
        clip = name[:5]
        if m.isfile() and name.endswith(".npz") and clip in CLIPS:
            m.name = str(Path(*Path(m.name).parts[-3:]))  # CMU/05/05_02_poses.npz
            tar.extract(m, out, filter="data")
            found.add(clip)
            print(f"  {m.name}  ({CLIPS[clip]})")

missing = set(CLIPS) - found
print(f"extracted {len(found)}/{len(CLIPS)} clips to {out}" + (f", missing {sorted(missing)}" if missing else ""))
