#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Dict, List
from nlf.activations import get_activation
from ..contract import contract_dict

from utils.ray_utils import (
    get_ray_density
)

from utils.intersect_utils import (
    sort_z,
    sort_with
)


def uniform_weight(cfg):
    def weight_fn(rays, dists):
        return torch.ones_like(dists)

    return weight_fn


def ease_max_weight(cfg):
    weight_start = cfg.weight_start if 'weight_start' in cfg else 1.0
    weight_end = cfg.weight_end if 'weight_end' in cfg else 0.95

    def weight_fn(rays, dists):
        rays_norm = torch.abs(nn.functional.normalize(rays[..., 3:6], p=float("inf"), dim=-1))
        weights = ((rays_norm - weight_end) / (weight_start - weight_end)).clamp(0, 1)
        return weights.unsqueeze(1).repeat(1, dists.shape[1] // 3, 1).view(
            weights.shape[0], -1
        )

    return weight_fn


weight_fn_dict = {
    'uniform': uniform_weight,
    'ease_max': ease_max_weight,
}


class Intersect(nn.Module):
    sort_outputs: List[str]

    def __init__(
        self,
        z_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cur_iter = 0
        self.cfg = cfg

        # Input/output size
        self.z_channels = z_channels
        self.in_density_field = cfg.in_density_field if 'in_density_field' in cfg else 'sigma'

        # Other common parameters
        self.forward_facing = cfg.forward_facing if 'forward_facing' in cfg else False
        self.normalize = cfg.normalize if 'normalize' in cfg else False
        self.residual_z = cfg.residual_z if 'residual_z' in cfg else False
        self.residual_distance = cfg.residual_distance if 'residual_distance' in cfg else False
        self.sort = cfg.sort if 'sort' in cfg else False
        self.sort_fixed = cfg.sort_fixed if 'sort_fixed' in cfg else False
        self.generate_offsets = cfg.generate_offsets if 'generate_offsets' in cfg else False
        self.clamp = cfg.clamp if 'clamp' in cfg else False

        self.use_dataset_bounds = cfg.use_dataset_bounds if 'use_dataset_bounds' in cfg else False
        self.use_disparity = cfg.use_disparity if 'use_disparity' in cfg else False
        self.use_sigma = cfg.use_sigma if 'use_sigma' in cfg else False

        # Origin
        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Minimum intersect distance
        if self.use_dataset_bounds:
            self.near = cfg.near if 'near' in cfg else kwargs['system'].dm.train_dataset.near
            self.far = cfg.far if 'far' in cfg else kwargs['system'].dm.train_dataset.far * 1.5
        else:
            self.near = cfg.near if 'near' in cfg else 0.0
            self.far = cfg.far if 'far' in cfg else float("inf")

        #self.near = 0.0 # TODO: Remove
        self.far = cfg.far if 'far' in cfg else float("inf")

        self.min_sep = cfg.min_sep if 'min_sep' in cfg else 0.0

        # Sorting
        self.weight_fn = weight_fn_dict[cfg.weight_fn.type](cfg.weight_fn) if 'weight_fn' in cfg else None
        self.sort_outputs = list(cfg.sort_outputs) if 'sort_outputs' in cfg else []

        if self.weight_fn is not None:
            self.sort_outputs.append('weights')

        # Mask
        if 'mask' in cfg:
            self.mask_stop_iters = cfg.mask.stop_iters if 'stop_iters' in cfg.mask else float("inf")
        else:
            self.mask_stop_iters = float("inf")

        # Contract function
        if 'contract' in cfg:
            self.contract_fn = contract_dict[cfg.contract.type](
                cfg.contract,
                **kwargs
            )
            self.contract_stop_iters = cfg.contract.stop_iters if 'stop_iters' in cfg.contract else float("inf")
        else:
            self.contract_fn = contract_dict['identity']({})
            self.contract_stop_iters = float("inf")

        # Activation
        self.activation = get_activation(cfg.activation if 'activation' in cfg else 'identity')

        # Dropout params
        self.use_dropout = 'dropout' in cfg
        self.dropout_frequency = cfg.dropout.frequency if 'dropout' in cfg else 2
        self.dropout_stop_iter = cfg.dropout.stop_iter if 'dropout' in cfg else float("inf")

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, self.z_scale.shape[-1]) * self.z_scale[None] + self.samples[None]
        z_vals = z_vals.view(z_vals.shape[0], -1)

        if self.contract_fn.contract_samples:
            z_vals = self.contract_fn.inverse_contract_distance(z_vals)
        elif self.use_disparity:
            z_vals = torch.where(
                torch.abs(z_vals) < 1e-8, 1e8 * torch.ones_like(z_vals), z_vals
            )
            z_vals = 1.0 / z_vals

        return z_vals
    
    def get_intersect_distances(self, rays, z_vals, x, render_kwargs):
        ## Z value processing
        z_vals = z_vals.view(rays.shape[0], -1)

        # Z activation and sigma
        if self.use_sigma and self.in_density_field in x:
            sigma = x[self.in_density_field].view(z_vals.shape[0], -1)
        else:
            sigma = torch.zeros(z_vals.shape[0], z_vals.shape[1], device=z_vals.device)

        z_vals = self.activation(z_vals.view(z_vals.shape[0], sigma.shape[1], -1)) * (1 - sigma.unsqueeze(-1))
        z_vals = z_vals.view(z_vals.shape[0], -1)

        # Apply offset
        if self.use_dropout and ((self.cur_iter % self.dropout_frequency) == 0) and self.cur_iter < self.dropout_stop_iter and self.training:
            z_vals = torch.zeros_like(z_vals)

        #print(torch.abs(z_vals).mean()) # TODO: Remove
        #print(torch.abs(z_vals.view(z_vals.shape[0], -1, 3)[..., -1]).mean()) # TODO: Remove

        # Add samples and contract
        z_vals = self.process_z_vals(z_vals)
        #print(z_vals.view(z_vals.shape[0], -1, 4)[0])

        # Residual distances
        if self.residual_z:
            if 'last_z' in x:
                last_z = x['last_z']
            elif 'last_distance' in x:
                last_distance = x['last_distance']
                #print(last_distance[0])
                last_z = self.distance_to_z(rays, last_distance)
                #print(last_z[0])

            z_vals = z_vals.view(z_vals.shape[0], last_z.shape[1], -1, last_z.shape[-1]) + last_z.unsqueeze(2)
            z_vals = z_vals.view(z_vals.shape[0], -1)

        # Get distances
        dists = self.intersect(rays, z_vals)

        # Calculate weights
        if self.weight_fn is not None:
            weights = self.weight_fn(rays, dists)
        else:
            weights = torch.ones_like(dists)

        if 'weights' not in x or x['weights'].shape[1] != weights.shape[1]:
            x['weights'] = weights.unsqueeze(-1)
        else:
            x['weights'] = x['weights'] * weights.unsqueeze(-1)

        # Mask
        mask = (dists <= self.near) | (dists >= self.far) | (weights == 0.0)

        if self.cur_iter > self.mask_stop_iters:
            mask = torch.zeros_like(mask)

        dists = torch.where(
            mask,
            torch.zeros_like(dists),
            dists
        )

        return dists, mask, z_vals

    def forward(self, rays: torch.Tensor, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = torch.cat(
            [
                rays[..., :3] - self.origin[None],
                rays[..., 3:6],
            ],
            dim=-1
        )

        # Get intersect distances, valid mask
        dists, mask, z_vals = self.get_intersect_distances(
            rays, x['z_vals'], x, render_kwargs
        )

        if self.sort_fixed or self.generate_offsets:
            dists_fixed, mask_fixed, z_vals_fixed = self.get_intersect_distances(
                rays, torch.zeros_like(x['z_vals']), x, render_kwargs
            )

        # Sort
        if self.sort:
            dists, sort_idx = sort_z(dists)

            if self.sort_fixed:
                dists_fixed, sort_idx = sort_z(dists_fixed)
            elif self.generate_offsets:
                dists_fixed = dists_fixed.unsqueeze(-1)
                dists_fixed = sort_with(sort_idx, dists_fixed)

            for output_key in self.sort_outputs:
                if output_key in x and len(x[output_key].shape) == 3:
                    x[output_key] = sort_with(sort_idx, x[output_key])

        # Mask again
        dists = dists.unsqueeze(-1)
        mask = (dists == 0.0)

        # Get points
        points = rays[..., None, :3] + rays[..., None, 3:6] * dists

        if self.generate_offsets:
            points_fixed = rays[..., None, :3] + rays[..., None, 3:6] * dists_fixed

        # Contract
        dists_no_contract = dists

        if not (self.cur_iter > self.contract_stop_iters):
            points, dists, origins = self.contract_fn.contract_points_and_distance(
                rays[..., :3],
                points,
                dists_no_contract
            )
            dists_no_contract = torch.where(
                mask,
                torch.zeros_like(dists_no_contract),
                dists_no_contract
            )

            if self.generate_offsets:
                points_fixed = self.contract_fn.contract_points(
                    points_fixed
                )

        # Return
        x['points'] = points
        x['origins'] = origins
        x['distances'] = dists
        x['distances_no_contract'] = dists_no_contract
        x['z_vals'] = z_vals

        if self.generate_offsets:
            x['point_offset_from_fixed'] = points - points_fixed

        return x
    
    def filter_min_separation(self, dists, points, x, render_kwargs):
        # Minimum separation
        if self.min_sep > 0.0:
            # Mask
            sep = dists[:, 1:] - dists[:, :-1]
            mask = sep < self.min_sep
            mask = torch.cat(
                [
                    torch.zeros_like(mask[:, :1]),
                    mask
                ],
                dim=1
            )

            # Mask distances, points
            dists = torch.where(mask, torch.zeros_like(dists), dists)
            dists_no_contract = torch.where(mask, torch.zeros_like(dists_no_contract), dists_no_contract)

            # Squeeze
            dists = dists.squeeze(-1)
            dists_no_contract = dists_no_contract.squeeze(-1)

            # Sort
            if self.sort:
                dists, sort_idx = sort_z(dists, 1, False)
                dists = dists.unsqueeze(-1)

                dists_no_contract = sort_with(sort_idx, dists_no_contract.unsqueeze(-1))
                points = sort_with(sort_idx, points)

                for output_key in self.sort_outputs:
                    if output_key in x and len(x[output_key].shape) == 3:
                        x[output_key] = sort_with(sort_idx, x[output_key])
        
        return dists, points
    
    def distance_to_z(self, rays, distance):
        pass

    def intersect(self, rays, z_vals):
        pass

    def set_iter(self, i):
        self.cur_iter = i
