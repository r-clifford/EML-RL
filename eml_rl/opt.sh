#!/bin/bash
LOG_BASE=$1
ALGO=$2
CONFIG=$3
# if paramater 4 is provided, use it as the study name
# otherwise, use the current timestamp
if [ -z "$4" ]; then
  STUDY_NAME="$ALGO-$(date '+%s')"
else
  STUDY_NAME="$4"
fi
echo "Study name: $STUDY_NAME"
LOG_DIR="$LOG_BASE/$STUDY_NAME"
OPT_DIR="$LOG_BASE/$STUDY_NAME"/opt

mkdir -p "$LOG_DIR"
cp eml_rl/reward.py "$LOG_DIR"
python rl-baselines3-zoo/train.py --algo "$ALGO" --env f1tenth-v0 --n-jobs 1 \
  -n 25000 \
  --eval-freq 2500 \
  --sampler tpe --pruner median \
  --conf-file "$CONFIG" --progress \
  -tb "$LOG_DIR" \
  -f "$LOG_DIR" \
  --optimization-log-path "$OPT_DIR" \
  -optimize --n-trials 500 \
  --save-freq 25000 \
  --n-evaluations 10 \
  --eval-episodes 10 \
  --seed 2024 \
  --storage "data/$ALGO.log" \
  --study-name "$STUDY_NAME" \
  --no-optim-plots \
  --max-total-trials 1000 \
  --uuid
# --storage "sqlite:///data/$ALGO.db" \
