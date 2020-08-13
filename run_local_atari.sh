# Copyright (c) Facebook, Inc. and its affiliates.

#!/bin/bash

GAME=AlienNoFrameskip-v4

CUDA_VISIBLE_DEVICES=0 python main.py \
  --env-name ${GAME} \
  --algo ppo_dice --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
  --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 \
  --use-linear-lr-decay --entropy-coef 0.01 --use-orthogonal-norm --disc-train 5
