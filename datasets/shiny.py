#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import cv2
from PIL import Image

import pytorch3d
import pytorch3d.io
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex
)
from matplotlib import pyplot as plt

from .llff import LLFFDataset

from utils.pose_utils import (
    interpolate_poses,
    correct_poses_bounds,
    create_spiral_poses,
)

from utils.ray_utils import (
    get_rays,
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
    to_ndc
)

from utils.intersect_utils import intersect_axis_plane


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class ShinyDataset(LLFFDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        self.faces_per_pixel = cfg.dataset.faces_per_pixel if 'faces_per_pixel' in cfg.dataset else 1
        self.use_depth = cfg.dataset.use_depth if 'use_depth' in cfg.dataset else False

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses_bounds.npy'),
            'rb'
        ) as f:
            poses_bounds = np.load(f)

        with self.pmgr.open(
            os.path.join(self.root_dir, 'hwf_cxcy.npy'),
            'rb'
        ) as f:
            hwfc = np.load(f)

        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )
        self.camera_ids = np.linspace(0, len(self.image_paths) - 1, len(self.image_paths))
        self.total_num_views = len(self.image_paths)

        self.dense = len(self.image_paths) > 80

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, 'images', image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :12].reshape(-1, 3, 4)
        self.bounds = poses_bounds[:, -2:]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = hwfc[:3, 0]
        self.cx, self.cy = hwfc[-2:, 0]

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75

        self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
            poses, self.bounds, use_train_pose=True
        )

        with self.pmgr.open(
            os.path.join(self.root_dir, 'planes.txt'),
            'r'
        ) as f:
            planes = [float(i) for i in f.read().strip().split(' ')]

        self.near = planes[0] * 0.95
        self.far = planes[1] * 1.05
        self.depth_range = np.array([self.near * 2.0, self.far])

        if self.dataset_cfg.collection in ['giants']:
            self.near = self.near * 0.8

        #self.near = self.bounds.min() * 0.95
        #self.far = self.bounds.max() * 1.05
        #self.depth_range = np.array([self.near * 2.0, self.far])

        #print(self.bounds.min())
        #print(np.median(self.bounds[..., 0]) * 0.5)
        #exit()

        # Load point cloud
        verts, faces = pytorch3d.io.load_ply(os.path.join(self.root_dir, 'dense/sparse/fused.ply'))

        verts = verts / scale_factor
        verts = torch.tensor(self.poses_avg[:3, :3]).float() @ verts.permute(1, 0) \
            + torch.tensor(self.poses_avg[:3, -1:]).float()
        verts = verts.permute(1, 0)

        if self.use_depth:
            meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
            self.image_size = (self.img_wh[1], self.img_wh[0])
            self.meshes = meshes.cuda()
            self.meshes.textures = TexturesVertex(torch.ones_like(self.meshes.verts_packed())[None])

        # Step 3: Ray directions for all pixels
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=self.centered_pixels
        )

        # Step 4: Holdout validation images
        if len(self.val_set) > 0:
            val_indices = self.val_set
        elif self.val_skip != 'inf':
            self.val_skip = min(
                len(self.image_paths), self.val_skip
            )
            val_indices = list(range(0, len(self.image_paths), self.val_skip))
        else:
            val_indices = []

        train_indices = [i for i in range(len(self.image_paths)) if i not in val_indices]

        if self.val_all:
            val_indices = [i for i in train_indices] # noqa

        train_indices = train_indices[::self.train_skip]

        if self.split == 'val' or self.split == 'test':
            self.image_paths = [self.image_paths[i] for i in val_indices]
            self.camera_ids = self.camera_ids[val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.camera_ids = self.camera_ids[train_indices]
            self.poses = self.poses[train_indices]
    
    def get_depth(self, idx):
        if not self.use_depth:
            return torch.zeros_like(self.directions.view(-1, 3)[..., 0:1].repeat(1, self.faces_per_pixel))

        # Convert pose
        pose = np.eye(4)
        pose[:3, :4] = self.poses[idx]
        pose[:3, -1:] = self.poses[idx][:3, -1:]
        pose[..., 2] *= -1
        pose[..., 0] *= -1
        pose = np.linalg.inv(pose)

        # Create pytorch3D camera
        R = torch.tensor(pose[:3, :3]).float()
        T = torch.tensor(pose[:3, -1:]).float()
        R = R.permute(1, 0)

        R = R[None]
        T = T[None, ..., 0]

        K = np.zeros((4, 4))
        K[:3, :3] = self.K
        K[2, 3] = 1.0
        K[2, 2] = 0.0
        K[3, 2] = 1.0
        K = torch.tensor(K)[None].float()

        cameras = PerspectiveCameras(
            device='cuda',
            R=R,
            T=T,
            K=K,
            image_size=(self.image_size,),
            in_ndc=False
        )

        # Render
        lights = PointLights(device='cuda', location=[[0.0, 0.0, -3.0]])
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=self.faces_per_pixel, 
            #max_faces_per_bin=1048576
        )

        renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device='cuda', 
                cameras=cameras,
                lights=lights
            )
        )

        _, depth = renderer(self.meshes)

        depth = depth[0].cpu().view(-1, self.faces_per_pixel)
        depth = depth * torch.linalg.norm(self.directions.view(-1, 3), dim=-1, keepdim=True)

        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], dim=-1)
        depth = to_ndc(depth, rays, -self.near)

        if self.use_ndc:
            ndc_rays = self.to_ndc(rays)
            max_depth = (1.0 - ndc_rays[..., 2:3]) / ndc_rays[..., 5:6]
        else:
            max_depth = (self.far - rays[..., 2:3]) / rays[..., 5:6]
        
        max_depth = max_depth.repeat(1, self.faces_per_pixel)

        depth[depth < 0.0] = max_depth[depth < 0.0]
        depth[depth > max_depth] = max_depth[depth > max_depth]

        #depth = np.uint8((depth / depth.max()) * 255.0)
        #cv2.imwrite('tmp_depth.png', depth)

        #rgb = self.get_rgb(idx)
        #rgb = rgb.reshape(image_size[0], image_size[1], 3).cpu().numpy()
        #rgb = np.uint8(rgb * 255.0)
        #cv2.imwrite('tmp_rgb.png', rgb)
        #exit()

        return depth
    
    def get_coords(self, idx):
        coords = super().get_coords(idx)
        depth = self.get_depth(idx)
        return torch.cat([coords, depth], dim=-1)

    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_depth = []
        self.all_rgb = []

        for idx in range(len(self.image_paths)):
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Depth
            self.all_depth += [self.get_depth(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

        self.update_all_data(
            torch.cat(self.all_coords, 0),
            torch.cat(self.all_rgb, 0),
            torch.cat(self.all_depth, 0),
        )

    def update_all_data(self, coords, rgb, depth):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_depth = depth
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
        batch['depth'] = batch['inputs'][..., self.all_coords.shape[-1] + 3:self.all_coords.shape[-1] + 4]
        batch['weight'] = batch['inputs'][..., -1:]
        del batch['inputs']

        return batch

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def prepare_render_data(self):
        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min()*.9, self.bounds.max()*5.

            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focus_depth = mean_dz

            if self.dense:
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses = create_spiral_poses(self.poses, radii, focus_depth * 4)
            else:
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                #radii[:2] = radii[:2] * 2.0
                self.poses = create_spiral_poses(self.poses, radii, focus_depth * 2)

            self.poses = np.stack(self.poses, axis=0)
            self.poses[..., :3, 3] = self.poses[..., :3, 3] - 0.1 * close_depth * self.poses[..., :3, 2]
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)


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


class DenseShinyDataset(ShinyDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        ## Bounds
        with self.pmgr.open(
            os.path.join(self.root_dir, 'bounds.npy'), 'rb'
        ) as f:
            bounds = np.load(f)

        self.bounds = bounds[:, -2:]

        ## Intrinsics
        with self.pmgr.open(
            os.path.join(self.root_dir, 'hwf_cxcy.npy'),
            'rb'
        ) as f:
            hwfc = np.load(f)

        ## Poses
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses.npy'), 'rb'
        ) as f:
            poses = np.load(f)

        ## Image paths
        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, 'images', image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        ## Skip
        row_skip = self.dataset_cfg.train_row_skip
        col_skip = self.dataset_cfg.train_col_skip

        poses_skipped = []
        image_paths_skipped = []

        for row in range(self.dataset_cfg.num_rows):
            for col in range(self.dataset_cfg.num_cols):
                idx = row * self.dataset_cfg.num_cols + col

                if self.split == 'train' and (
                    (row % row_skip) != 0 or (col % col_skip) != 0 or (idx % self.val_skip) == 0
                    ):
                    continue

                if (self.split == 'val' or self.split == 'test') and (
                    ((row % row_skip) == 0 and (col % col_skip) == 0) and (idx % self.val_skip) != 0
                    ):
                    continue

                poses_skipped.append(poses[idx])
                image_paths_skipped.append(self.image_paths[idx])

        poses = np.stack(poses_skipped, axis=0)
        self.poses = poses.reshape(-1, 3, 5)
        self.image_paths = image_paths_skipped

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = hwfc[:3, 0]
        self.cx, self.cy = hwfc[-2:, 0]

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.near = self.bounds.min()
        self.far = self.bounds.max()

        # Step 3: Ray directions for all pixels
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=self.centered_pixels
        )
