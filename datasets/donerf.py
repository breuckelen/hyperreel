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


class DONeRFDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        self.correct_poses = cfg.dataset.correct_poses if 'correct_poses' in cfg.dataset else False
        self.center_poses = cfg.dataset.center_poses if 'center_poses' in cfg.dataset else False
        self.use_ndc = cfg.dataset.use_ndc if 'use_ndc' in cfg.dataset else False
        self.train_skip = cfg.dataset.train_skip if 'train_skip' in cfg.dataset else 1
        self.depth_downsample = cfg.dataset.depth_downsample if 'depth_downsample' in cfg.dataset else None
        self.layered_depth = cfg.dataset.layered_depth if 'layered_depth' in cfg.dataset else False

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        if self.split == 'render':
            self.read_meta_for_split('test', 'cam_path_pan.json')
        elif self.split == 'test':
            self.read_meta_for_split('test', 'transforms_test.json')
        elif self.split == 'train':
            self.read_meta_for_split('train', 'transforms_train.json')
        elif self.split == 'val':
            self.read_meta_for_split('val', 'transforms_val.json')
        else:
            self.read_meta_for_split(self.split, 'transforms_test.json')

    def load_poses_from_meta(self, meta, dataset_meta):
        origin = np.array(dataset_meta['view_cell_center'])

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
            pose = np.array(frame['transform_matrix'])[:3, :4]

            if self.center_poses:
                pose[:3, -1] = pose[:3, -1] - origin

            poses += [pose]

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
        elif split == 'train':
            self.meta['frames'] = self.meta['frames'][::self.train_skip]

        # Load dataset info
        with self.pmgr.open(
            os.path.join(self.root_dir, 'dataset_info.json'),
            'r'
        ) as f:
            self.dataset_meta = json.load(f)

        W, H = self.img_wh

        self.focal = 0.5 * 800 / np.tan(
            0.5 * self.dataset_meta['camera_angle_x']
        )
        self.focal *= self.img_wh[0] / 800

        self.K = np.eye(3)
        self.K[0, 0] = self.focal
        self.K[0, 2] = (W / 2.0)
        self.K[1, 1] = self.focal
        self.K[1, 2] = (H / 2.0)

        # Bounds, common for all scenes
        self.depth_range = self.dataset_meta['depth_range']
        self.near = self.dataset_meta['depth_range'][0]
        self.far = self.dataset_meta['depth_range'][1]
        #self.depth_range = np.array([self.near * 1.5, self.far])

        self.view_cell_size = np.max(np.array(self.dataset_meta['view_cell_size']))
        self.bounds = np.array([self.near, self.far])

        # Image paths and pose
        self.train_poses, _ = self.load_poses_from_meta(self.train_meta, self.dataset_meta)
        self.poses, self.image_paths = self.load_poses_from_meta(self.meta, self.dataset_meta)

        # Correct
        if self.use_ndc or self.correct_poses:
            self.poses, _ = center_poses_with_rotation_only(self.poses, self.train_poses)
        
            if self.split == "render":
                if self.dataset_cfg.collection in ['pavillon']:
                    self.poses[..., :3, -1] *= 1.0
                else:
                    self.poses[..., :3, -1] *= 2.0

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
        self.all_depth = []
        self.all_points = []

        for idx in range(len(self.image_paths)):
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

            # Depth
            self.all_depth += [self.get_depth(idx)]

            # Points
            self.all_points += [self.get_points(idx)]

        self.update_all_data(
            torch.cat(self.all_coords, 0),
            torch.cat(self.all_rgb, 0),
            torch.cat(self.all_depth, 0),
            torch.cat(self.all_points, 0),
        )

        # Calculate bounds
        if self.layered_depth:
            mask = (self.all_depth[..., -1:] != 0.0)
        else:
            mask = (self.all_depth != 0.0)

        self.bbox_min = self.all_points[mask.repeat(1, 3)].reshape(-1, 3).min(0)[0]
        self.bbox_max = self.all_points[mask.repeat(1, 3)].reshape(-1, 3).max(0)[0]

        #self.bbox_min = self.all_points[mask.repeat(1, 3)].reshape(-1, 3).max(0)[0]
        #self.bbox_max = self.all_points[mask.repeat(1, 3)].reshape(-1, 3).min(0)[0]

        #self.near = float(self.all_depth[mask].min())
        #self.far = float(self.all_depth[mask].max())

    def update_all_data(self, coords, rgb, depth, points):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_depth = depth
        self.all_points = points
        self.all_weights = self.get_weights()

        ## Patches
        if self.use_patches or self.use_crop:
            self._all_coords = torch.clone(self.all_coords)
            self._all_rgb = torch.clone(self.all_rgb)
            self._all_depth = torch.clone(self.all_depth)

        ## All inputs
        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_depth, self.all_weights], -1
        )

    def format_batch(self, batch):
        batch['coords'] = batch['inputs'][..., :self.all_coords.shape[-1]]
        batch['rgb'] = batch['inputs'][..., self.all_coords.shape[-1]:self.all_coords.shape[-1] + 3]
        if self.layered_depth:
            batch['depth'] = batch['inputs'][..., self.all_coords.shape[-1] + 3:self.all_coords.shape[-1] + 5]
        else:
            batch['depth'] = batch['inputs'][..., self.all_coords.shape[-1] + 3:self.all_coords.shape[-1] + 4]
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
            coords = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            coords = torch.cat([rays_o, rays_d], dim=-1)
        
        depth = self.get_depth(idx)

        return torch.cat([coords, depth], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
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

    def get_depth(self, idx):
        image_path = self.image_paths[idx]
        image_path = os.path.join(self.root_dir, f'{image_path}_depth.npz')

        if not self.pmgr.exists(image_path):
            if self.layered_depth:
                return torch.zeros_like(self.directions.view(-1, 3)[..., 0:2])
            else:
                return torch.zeros_like(self.directions.view(-1, 3)[..., 0:1])

        with self.pmgr.open(
            image_path,
            'rb'
        ) as depth_file:
            with np.load(depth_file) as depth:
                depth = depth['arr_0'].reshape(800, 800)

        # Resize
        if self.depth_downsample != None:
            img_wh = (self._img_wh[0] // self.depth_downsample, self._img_wh[1] // self.depth_downsample)
            depth = cv2.resize(depth, img_wh, interpolation=cv2.INTER_NEAREST)
            depth[depth < self.near] = self.near
            depth[depth > self.far] = self.far
        
        if self.layered_depth:
            kernel = np.ones((5,5),np.uint8)
            depth_min = cv2.erode(depth, kernel)
            depth_max = cv2.dilate(depth, kernel)
            depth = np.stack([depth_min, depth_max], axis=-1)

        depth = cv2.resize(depth, self._img_wh, interpolation=cv2.INTER_NEAREST)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            depth = cv2.resize(depth, self.img_wh, interpolation=cv2.INTER_NEAREST)

        # Flip
        depth = np.flip(depth, 0)

        # Depth to distance
        if self.layered_depth:
            depth = depth * np.array(torch.linalg.norm(self.directions, dim=-1))[..., None]
        else:
            depth = depth * np.array(torch.linalg.norm(self.directions, dim=-1))

        # Change zero values
        depth[depth < self.near] = self.near
        depth[depth > self.far] = self.far

        # Transform
        depth = self.transform(np.copy(depth)).permute(1, 2, 0)

        # Return
        if self.layered_depth:
            return depth.view(-1, 2)
        else:
            return depth.view(-1, 1)

    def get_points(self, idx):
        rays = self.all_coords[idx][..., :6].reshape(-1, 6)

        if self.layered_depth:
            depth = self.all_depth[idx].reshape(-1, 2)[..., -1:]
            return rays[..., :3] + rays[..., 3:6] * depth
        else:
            depth = self.all_depth[idx].reshape(-1, 1)
            return rays[..., :3] + rays[..., 3:6] * depth

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal
        K[0, 2] = self.img_wh[0] / 2
        K[1, 1] = self.focal
        K[1, 2] = self.img_wh[1] / 2

        return K

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
                'depth': self.get_depth(idx),
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
