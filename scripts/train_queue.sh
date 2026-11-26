#!/bin/bash
# Train RL tracking policies on several clips one after another (each run in its own folder, tag "q").
# Usage: nohup caffeinate -i bash scripts/train_queue.sh > outputs/tracking/queue.log 2>&1 &
cd "$(dirname "$0")/.."
run() { echo "=== $(date +%H:%M) start $1 0-$2 s, $3 steps"; .venv/bin/python scripts/train_tracking.py --clip "$1" --t0 0 --t1 "$2" --steps "$3" --tag q > "outputs/tracking/train_$1_q.log" 2>&1; echo "=== $(date +%H:%M) done $1 (exit $?)"; }
run 02_01_poses 2.8 8e6     # walk
run 07_01_poses 2.6 8e6     # walk
run 05_06_poses 7.3 15e6    # modern dance: cartwheel-like start, pirouettes, jete
run 60_01_poses 37.3 30e6   # salsa (60 fps source)
echo "=== $(date +%H:%M) queue finished"
