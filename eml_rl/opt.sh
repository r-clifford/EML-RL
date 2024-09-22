#!/bin/bash
LOG_BASE=$1
ALGO=$2
CONFIG=$3
STUDY_NAME="$ALGO-$(date '+%s')"
LOG_DIR="$LOG_BASE$STUDY_NAME"
OPT_DIR="$LOG_BASE$STUDY_NAME"/opt

mkdir -p "$LOG_DIR"
cp eml_rl/reward.py "$LOG_DIR"
python rl-baselines3-zoo/train.py --algo "$ALGO" --env f1tenth-v0 --n-jobs 1 \
  -n 50000 \
  --eval-freq 5000 \
  --sampler tpe --pruner median \
  --conf-file "$CONFIG" --progress \
  -tb "$LOG_DIR" \
  -f "$LOG_DIR" \
  --optimization-log-path "$OPT_DIR" \
  -optimize --n-trials 500 \
  --save-freq 25000 \
  --n-evaluations 10 \
  --eval-episodes 20 \
  --seed 2024 \
  --storage "sqlite:///studies.db" \
  --study-name "$STUDY_NAME" \
  --uuid
