"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
from typing import Callable, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.tensorf_utils import (
    raw2alpha,
    alpha2weights,
    scale_shift_color_all,
    scale_shift_color_one,
    transform_color_all,
    transform_color_one
)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.opt_group = "color"

        self.distance_scale = cfg.distance_scale if 'distance_scale' in cfg else 1.0
        self.white_bg = cfg.white_bg if 'white_bg' in cfg else 0
        self.black_bg = cfg.black_bg if 'black_bg' in cfg else 0

        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)

        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=cfg.net_depth,
            net_width=cfg.net_width,
            skip_layer=cfg.skip_layer,
            net_depth_condition=cfg.net_depth_condition,
            net_width_condition=cfg.net_width_condition,
        )

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def _forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)

    def set_iter(self, i):
        pass

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


nerf_base_dict = {
    "nerf_static": VanillaNeRFRadianceField,
}