#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import cv2

import numpy as np
import torch

from PIL import Image

from utils.pose_utils import (
    correct_poses_bounds,
    center_poses_with,
    center_poses_with_rotation_only,
    create_spiral_poses,
    create_rotating_spiral_poses,
    interpolate_poses,
)

from utils.ray_utils import (
    get_ndc_rays_fx_fy,
    get_ray_directions_K,
    get_rays
)

from .base import Base5DDataset, Base6DDataset
from .lightfield import LightfieldDataset


class EikonalDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        self.correct_poses = cfg.dataset.correct_poses if 'correct_poses' in cfg.dataset else False
        self.center_poses = cfg.dataset.center_poses if 'center_poses' in cfg.dataset else False
        self.use_ndc = cfg.dataset.use_ndc if 'use_ndc' in cfg.dataset else False

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        if self.split == 'render':
            self.read_meta_for_split('test', 'transforms_test_video.json')
        elif self.split == 'test':
            self.read_meta_for_split('test', 'transforms_test.json')
        elif self.split == 'train':
            self.read_meta_for_split('train', 'transforms_train.json')
        elif self.split == 'val':
            self.read_meta_for_split('val', 'transforms_val.json')
        else:
            self.read_meta_for_split(self.split, 'transforms_test.json')

    def load_poses_from_meta(self, meta):
        # Image paths and pose
        image_paths = []
        poses = []

        for frame in meta['frames']:
            # Image path
            if 'file_path' in frame:
                image_paths += [frame['file_path']]
            else:
                image_paths += [None]

            # Pose
            pose = np.array(frame['transform_matrix'])[:4, :4]
            pose_pre = np.eye(4)
            pose_pre[1, 1] *= -1
            pose_pre[2, 2] *= -1

            pose = pose_pre @ pose @ pose_pre
            poses += [pose[:3, :4]]

        poses = np.stack(poses, axis=0)

        return poses, image_paths

    def read_meta_for_split(self, split, split_file):
        # Load train meta
        with self.pmgr.open(
            os.path.join(self.root_dir, 'transforms_train.json'),
            'r'
        ) as f:
            self.train_meta = json.load(f)

        # Load meta
        with self.pmgr.open(
            os.path.join(self.root_dir, split_file),
            'r'
        ) as f:
            self.meta = json.load(f)

        if split == 'val':
            self.meta['frames'] = self.meta['frames'][:self.val_num]
        
        W, H = self.img_wh
        self.K = np.array(self.meta['cam_mat'])

        if self.dataset_cfg.collection in ['pen', 'glass']:
            self.depth_range = [0.2, 14.0]
        elif self.dataset_cfg.collection in ['ball']:
            self.depth_range = [0.2, 12.0]
        elif self.dataset_cfg.collection in ['dolphin']:
            self.depth_range = [0.2, 1.2]

        self.near = self.depth_range[0]
        self.far = self.depth_range[1]
        self.bounds = np.array([self.near, self.far])

        # Image paths and pose
        self.train_poses, _ = self.load_poses_from_meta(self.train_meta)
        self.poses, self.image_paths = self.load_poses_from_meta(self.meta)

        # Correct
        if self.use_ndc or self.correct_poses:
            self.poses, _ = center_poses_with(self.poses, self.train_poses)

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            H, W, self.K, centered_pixels=self.centered_pixels
        )

    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []

        for idx in range(len(self.image_paths)):
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

        self.update_all_data(
            torch.cat(self.all_coords, 0),
            torch.cat(self.all_rgb, 0),
        )

    def update_all_data(self, coords, rgb):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_weights = self.get_weights()

        ## Patches
        if self.use_patches or self.use_crop:
            self._all_coords = torch.clone(self.all_coords)
            self._all_rgb = torch.clone(self.all_rgb)

        ## All inputs
        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_weights], -1
        )

    def format_batch(self, batch):
        batch['coords'] = batch['inputs'][..., :self.all_coords.shape[-1]]
        batch['rgb'] = batch['inputs'][..., self.all_coords.shape[-1]:self.all_coords.shape[-1] + 3]
        batch['weight'] = batch['inputs'][..., -1:]
        del batch['inputs']

        return batch

    def prepare_render_data(self):
        self.prepare_test_data()

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)

        if self.use_ndc:
            return self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}'),
            'rb'
        ) as im_file:
            img = np.array(Image.open(im_file).convert('RGBA'))

        img = cv2.resize(img, self._img_wh, interpolation=cv2.INTER_AREA)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_AREA)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        return np.copy(self.K)

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'coords': self.get_coords(idx),
                'pose': self.poses[idx],
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])

        elif self.split == 'test':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        elif self.split == 'val':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }


        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch

