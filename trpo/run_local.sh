# Copyright (c) Facebook, Inc. and its affiliates.

#!/bin/bash

GAME=Reacher-v2

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --env-name "GAME" \
  --log-dir results
