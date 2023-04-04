#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=donerf_large \
    experiment/training=donerf_tensorf_reg \
    experiment.training.val_every=5 \
    experiment.training.ckpt_every=20 \
    experiment.training.render_every=10 \
    experiment.training.test_every=20 \
    experiment.training.num_epochs=21 \
    experiment/model=donerf$4 \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    experiment.dataset.train_skip=$3 \
    +experiment/regularizers/feedback=point_offset \
    +experiment/regularizers/tensorf=tv_4000 \
    +experiment/visualizers/embedding=default \
    experiment.params.name=donerf_$2_skip_$3$4