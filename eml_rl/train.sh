#!/bin/bash
LOG_BASE=$1
ALGO=$2
CONFIG=$3
echo "LOG_BASE: $LOG_BASE"
echo "ALGO: $ALGO"
LOG_DIR="$LOG_BASE$ALGO-$(date '+%s')"
echo "LOG_DIR: $LOG_DIR"
tensorboard --logdir "$LOG_DIR" &
TB_PID=$!
mkdir -p "$LOG_DIR"
cp eml_rl/reward.py "$LOG_DIR"
python rl-baselines3-zoo/train.py --algo "$ALGO" --env f1tenth-v0 --n-jobs 1 \
  -n 1000000 \
  --eval-freq 25000 \
  --conf-file "$CONFIG" --progress \
  -tb "$LOG_DIR" \
  -f "$LOG_DIR" \
  --save-freq 25000 \
  --eval-episodes 20 \
  --seed 2024 \
  --uuid

kill $TB_PID
