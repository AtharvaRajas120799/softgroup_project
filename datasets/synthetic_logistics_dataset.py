import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticLogisticsDataset(Dataset):
    """
    Loads one scene at a time from your synthetic dataset:
    - xyz: (N,3)
    - semantic labels: (N,)
    - instance labels: (N,)
    - rgb: (N,4)  -> keep first 3, drop alpha
    - normals: (N,4) -> keep first 3, drop homogeneous
    Returns a dict that SoftGroup-style pipelines use.
    """

    def __init__(self, root_dir, scene_ids, use_rgb=True, use_normals=True, rgb_normalize=True):
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.use_rgb = use_rgb
        self.use_normals = use_normals
        self.rgb_normalize = rgb_normalize

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]

        # Load arrays
        xyz = np.load(os.path.join(self.root_dir, f"{sid}_pointcloud.npy"))  # (N,3)
        semantic = np.load(os.path.join(self.root_dir, f"{sid}_pointcloud_semantic.npy"))  # (N,)
        instance = np.load(os.path.join(self.root_dir, f"{sid}_pointcloud_instance.npy"))  # (N,)
        rgb = np.load(os.path.join(self.root_dir, f"{sid}_pointcloud_rgb.npy"))  # (N,4)
        normals = np.load(os.path.join(self.root_dir, f"{sid}_pointcloud_normals.npy"))  # (N,4)

        # Clean channels
        if self.use_rgb:
            rgb = rgb[:, :3]  # drop alpha (constant 255)
            if self.rgb_normalize:
                rgb = rgb.astype(np.float32) / 255.0

        if self.use_normals:
            normals = normals[:, :3].astype(np.float32)  # drop homogeneous (constant 1)

        # Build features
        feat_list = []
        if self.use_rgb:
            feat_list.append(rgb)
        if self.use_normals:
            feat_list.append(normals)

        feat = None
        if len(feat_list) > 0:
            feat = np.concatenate(feat_list, axis=1)  # (N,C)

        # Convert to torch tensors
        coord = torch.from_numpy(xyz).float()
        semantic_label = torch.from_numpy(semantic).long()
        instance_label = torch.from_numpy(instance).long()

        if feat is not None:
            feat = torch.from_numpy(feat).float()

        return {
            "coord": coord,                     # (N,3)
            "feat": feat,                       # (N,C) or None
            "semantic_label": semantic_label,   # (N,)
            "instance_label": instance_label    # (N,)
        }
