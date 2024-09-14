#!/bin/bash
cwd=$(dirname "$0")
# tensorboard --logdir "$cwd"/logs/sac-log &
python rl-baselines3-zoo/train.py --algo td3 --env f1tenth-v0 --n-jobs 1 \
-n 100000 \
--eval-freq 25000 \
 --sampler tpe --pruner median \
 --conf-file eml_rl/config/hyperparams/td3_f1tenth.py --progress -tb td3-log\
 -optimize --n-trials 500 \
 --save-freq 50000

# python -m rl_zoo3.train --algo sac --env f1tenth-v0  --n-jobs 1 \
#  --sampler tpe --pruner median \
#  -n 2000000 \
#  --eval-freq 10000 \
#  --conf-file "$cwd"/config/hyperparams/sac_f1tenth.py --progress -tb "$cwd"/logs/sac-log \
#  --optimization-log-path "$cwd"/logs/sac-opt \
#  -optimize 
