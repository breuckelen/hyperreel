#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

LD_LIBRARY_PATH="" bash scripts/run_one_donerf.sh $1 barbershop $2 $3
LD_LIBRARY_PATH="" bash scripts/run_one_donerf.sh $1 pavillon $2 $3
LD_LIBRARY_PATH="" bash scripts/run_one_donerf.sh $1 bulldozer $2 $3
LD_LIBRARY_PATH="" bash scripts/run_one_donerf.sh $1 classroom $2 $3
LD_LIBRARY_PATH="" bash scripts/run_one_donerf.sh $1 forest $2 $3
LD_LIBRARY_PATH="" bash scripts/run_one_donerf.sh $1 sanmiguel $2 $3