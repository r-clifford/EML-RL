#!/bin/bash
cwd=$(dirname "$0")
# tensorboard --logdir "$cwd"/logs/sac-log &
# python train.py --algo td3 --env f110-v0 --n-jobs 1 \
# -n 1000000 \
#  --sampler tpe --pruner median \
#  --conf-file hyperparams/python/td3_f1tenth.py --progress -tb td3-log\
#  -optimize \

python -m rl_zoo3.train --algo sac --env f1tenth-v0  --n-jobs 1 \
 --sampler tpe --pruner median \
 -n 2000000 \
 --eval-freq 10000 \
 --conf-file "$cwd"/config/hyperparams/sac_f1tenth.py --progress -tb "$cwd"/logs/sac-log \
 --optimization-log-path "$cwd"/logs/sac-opt \
#  -optimize 
