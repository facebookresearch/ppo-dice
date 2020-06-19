# Copyright (c) Facebook, Inc. and its affiliates.

#!/bin/bash

GAME=HalfCheetah-v2

CUDA_VISIBLE_DEVICES=0 python main.py \
  --env-name ${GAME} \
  --algo ppo_dice \
   --use-gae \
  --log-interval 1 \
  --num-steps 2048 \
  --num-processes 1 \
  --lr 3e-4 \
  --use-orthogonal-norm \
  --entropy-coef 0 \
  --value-loss-coef 0.5 \
  --ppo-epoch 10 \
  --num-mini-batch 32 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --log-dir logs \
  --num-env-steps 1000000 \
  --use-linear-lr-decay \
  --use-proper-time-limits
