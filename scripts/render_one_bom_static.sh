#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=blender_open_movies_static \
    experiment/training=bom_tensorf \
    experiment.training.val_every=5 \
    experiment.training.render_every=10 \
    experiment.training.test_every=40 \
    experiment.training.ckpt_every=10 \
    experiment.training.num_epochs=40 \
    experiment/model=bom_sphere_static \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.num_frames=1 \
    experiment.dataset.lightfield_step=$3 \
    experiment.params.name=bom_static_$2_step_$3 \
    experiment.params.save_results=True \
    experiment.training.num_iters=100 \
    experiment.training.num_epochs=1000 \
    experiment.params.render_only=True

