import time

import numpy as np
import torch
import trimesh

from src.utils import mc_from_psr


class Generator3D(object):
    """Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        device (device): pytorch device
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
    """

    def __init__(
        self,
        model,
        points_batch_size=100000,
        threshold=0.5,
        device=None,
        padding=0.1,
        sample=False,
        input_type=None,
        dpsr=None,
        psr_tanh=True,
    ):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.device = device
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.dpsr = dpsr
        self.psr_tanh = psr_tanh

    def generate_mesh(self, data, return_stats=False, progress_callback=None):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
            progress_callback: Optional callable(step_name, step_num, total_steps) for progress reporting
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        p = data.to(device)

        if progress_callback:
            progress_callback("Encoding point cloud", 1, 3)

        t0 = time.time()
        points, normals = self.model(p)
        t1 = time.time()

        if progress_callback:
            progress_callback("Running DPSR", 2, 3)

        psr_grid = self.dpsr(points, normals)
        t2 = time.time()

        if progress_callback:
            progress_callback("Extracting mesh (marching cubes)", 3, 3)

        v, f, _ = mc_from_psr(psr_grid, zero_level=self.threshold)
        stats_dict["pcl"] = t1 - t0
        stats_dict["dpsr"] = t2 - t1
        stats_dict["mc"] = time.time() - t2
        stats_dict["total"] = time.time() - t0

        if return_stats:
            return v, f, points, normals, psr_grid, stats_dict
        else:
            return v, f, points, normals, psr_grid
