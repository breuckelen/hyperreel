#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=stanford_large \
    experiment/training=stanford_nerf \
    experiment.training.val_every=1 \
    experiment.training.ckpt_every=20 \
    experiment.training.test_every=20 \
    experiment.training.render_every=50 \
    ++experiment.training.num_epochs=100 \
    experiment/model=stanford_z_plane_nerf \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    experiment.dataset.lightfield.step=$3 \
    experiment.params.name=stanford_$2_step_$3_nerf



