"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import math
from typing import Callable, List, Union, Dict

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn

from utils.tensorf_utils import (
    raw2alpha,
    alpha2weights,
    scale_shift_color_all,
    scale_shift_color_one,
    transform_color_all,
    transform_color_one
)


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.distance_scale = cfg.distance_scale if 'distance_scale' in cfg else 1.0
        self.white_bg = cfg.white_bg if 'white_bg' in cfg else 0
        self.black_bg = cfg.black_bg if 'black_bg' in cfg else 0

        self.register_buffer('aabb', torch.tensor(cfg.aabb))

        self.num_dim = cfg.num_dim
        self.n_levels = cfg.n_levels
        self.use_viewdirs = cfg.use_viewdirs

        # NOTE: Density activation is new
        if 'density_activation' not in cfg:
            self.density_activation = lambda x: trunc_exp(x - 1)
        elif cfg.density_activation == 'trunc_exp':
            self.density_activation = lambda x: trunc_exp(x - 1)
        elif cfg.density_activation == 'relu':
            self.density_activation = lambda x: torch.relu(x)

        self.geo_feat_dim = cfg.geo_feat_dim

        per_level_scale = math.exp(
            (math.log(cfg.max_res) - math.log(cfg.base_res)) / (cfg.n_levels - 1)
        )

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=cfg.num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )
        
        # NOTE: Splitting network into Encoding and MLP is new
        self.grid_encoding = tcnn.Encoding(
            n_input_dims=cfg.num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": cfg.n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": cfg.log2_hashmap_size,
                "base_resolution": cfg.base_res,
                "per_level_scale": per_level_scale,
            },
        )

        self.mlp_base = tcnn.Network(
            n_input_dims=self.grid_encoding.n_output_dims,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": cfg.hidden_dim,
                "n_hidden_layers": 1,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=(
                (
                    self.direction_encoding.n_output_dims
                    if self.use_viewdirs
                    else 0
                )
                + self.geo_feat_dim
            ),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        # NOTE: Windowing is new
        self.cur_iter = 0
        self.window_iters = cfg.window_iters if "window_iters" in cfg else 0.0

        if self.window_iters > 0:
            self.n_window_levels = cfg.n_window_levels
            self.window_iters_single = self.window_iters // self.n_window_levels
            self.window_iters_list = np.arange(0, self.window_iters, self.window_iters_single)

        # NOTE: Different learning rates for encoding and mlps is new
        #self.opt_group = 'color'
        self.opt_group = {
            "color": [
                self.grid_encoding,
            ],
            "color_impl": [
                self.mlp_base,
                self.mlp_head,
            ],
        }
    
    def window_features(self, feat):
        if self.window_iters == 0 or self.n_window_levels == 0 \
            or self.cur_iter >= self.window_iters_list[-1]:
            return feat

        feat_split = torch.split(feat, 2 * self.n_levels // self.n_window_levels , dim=-1)
        new_features = [feat_split[0]]

        for l, cur_feat in enumerate(feat_split[1:]):
            window_start = self.window_iters_list[l+1]
            window_end = window_start + self.window_iters_single

            if self.cur_iter >= window_end:
                w = 1.0
            elif self.cur_iter <= window_start:
                w = 0.0
            else:
                w = min(max(float(self.cur_iter - window_start) / self.window_iters_single, 0.0), 1.0)
                #w = 1.0
            
            new_features.append(w * cur_feat)
        
        return torch.cat(new_features, dim=-1)

    def query_density(self, x, return_feat: bool = False):
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        # Get features
        feat = self.grid_encoding(x.view(-1, self.num_dim))
        feat = self.window_features(feat)

        # Run base MLP
        x = (
            self.mlp_base(feat)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )

        # Get density
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )

        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )

        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.view(-1, self.geo_feat_dim)

        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )

        return rgb

    def _forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density
    
    def set_iter(self, i):
        self.cur_iter = i

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        # Batch size
        batch_size = x["viewdirs"].shape[0]

        # Positions
        nSamples = x["points"].shape[-1] // 3
        xyz_sampled = x["points"].view(batch_size, nSamples, 3)

        # Distances
        distances = x["distances"].view(batch_size, -1)
        deltas = torch.cat(
            [
                distances[..., 1:] - distances[..., :-1],
                1e10 * torch.ones_like(distances[:, :1]),
            ],
            dim=1,
        )

        # Viewdirs
        viewdirs = x["viewdirs"].view(batch_size, nSamples, 3)

        # Weights
        weights = x["weights"].view(batch_size, -1, 1)

        # Mask out
        ray_valid = distances > 0
    
        # Get densities and colors
        sigma = xyz_sampled.new_zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = xyz_sampled.new_zeros(
            (xyz_sampled.shape[0], xyz_sampled.shape[1], 3), device=xyz_sampled.device
        )

        if ray_valid.any():
            valid_rgbs, valid_sigma = self._forward(
                xyz_sampled[ray_valid],
                viewdirs[ray_valid]
            )

            sigma[ray_valid] = valid_sigma.view(-1)
            rgb[ray_valid] = valid_rgbs.view(-1, 3)

        alpha, weight, bg_weight = raw2alpha(sigma, deltas * self.distance_scale)

        # Transform colors
        if 'color_scale' in x:
            color_scale = x['color_scale'].view(rgb.shape[0], rgb.shape[1], 3)
            color_shift = x['color_shift'].view(rgb.shape[0], rgb.shape[1], 3)
            rgb = scale_shift_color_all(rgb, color_scale, color_shift)
        elif 'color_transform' in x:
            color_transform = x['color_transform'].view(rgb.shape[0], rgb.shape[1], 9)
            color_shift = x['color_shift'].view(rgb.shape[0], rgb.shape[1], 3)
            rgb = transform_color_all(rgb, color_transform, color_shift)

        # Over composite
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[:, :, None] * rgb, -2)

        # White background
        if (self.white_bg or (self.training and torch.rand((1,)) < 0.5)) and not self.black_bg:
            rgb_map = rgb_map + (1.0 - acc_map[:, None])

        # Transform colors
        if 'color_scale_global' in x:
            rgb_map = scale_shift_color_one(rgb, rgb_map, x)
        elif 'color_transform_global' in x:
            rgb_map = transform_color_one(rgb, rgb_map, x)

        # Clamp and return
        if not self.training:
            rgb_map = rgb_map.clamp(0, 1)

        # Other fields
        outputs = {
            "rgb": rgb_map
        }

        fields = render_kwargs.get("fields", [])
        no_over_fields = render_kwargs.get("no_over_fields", [])
        pred_weights_fields = render_kwargs.get("pred_weights_fields", [])

        if len(fields) == 0:
            return outputs

        if len(pred_weights_fields) > 0:
            pred_weights = alpha2weights(weights[..., 0])

        for key in fields:
            if key == 'render_weights':
                outputs[key] = weight
            elif key in no_over_fields:
                outputs[key] = x[key].view(batch_size, -1)
            elif key in pred_weights_fields:
                outputs[key] = torch.sum(
                    pred_weights[..., None] * x[key].view(batch_size, nSamples, -1),
                    -2,
                )
            else:
                outputs[key] = torch.sum(
                    weight[..., None] * x[key].view(batch_size, nSamples, -1),
                    -2,
                )

        return outputs


ngp_base_dict = {
    "ngp_static": NGPRadianceField
}