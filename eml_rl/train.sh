#!/bin/bash
LOG_BASE=$1
ALGO=$2
CONFIG=$3
echo "LOG_BASE: $LOG_BASE"
echo "ALGO: $ALGO"
LOG_DIR="$LOG_BASE/$ALGO-$(date '+%s')"
echo "LOG_DIR: $LOG_DIR"
# tensorboard --logdir "$LOG_DIR" &
tensorboard --logdir "$LOG_BASE" &
TB_PID=$!
mkdir -p "$LOG_DIR"
zip -r eml_rl.zip eml_rl
cp eml_rl.zip "$LOG_DIR"
python rl-baselines3-zoo/train.py --algo "$ALGO" --env f1tenth-v0 \
	-n 2000000 \
	--eval-freq 10000 \
	--conf-file "$CONFIG" --progress \
	-tb "$LOG_DIR" \
	-f "$LOG_DIR" \
	--save-freq 10000 \
	--eval-episodes 16 \
	--seed 2025 \
	--uuid

kill $TB_PID
