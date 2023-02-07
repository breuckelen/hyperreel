#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=eikonal \
    experiment/training=donerf_tensorf \
    experiment.training.val_every=5 \
    experiment.training.test_every=10 \
    experiment.training.render_every=50 \
    experiment.training.ckpt_every=80 \
    experiment/model=eikonal_sphere \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.params.name=eikonal_$2_sphere



