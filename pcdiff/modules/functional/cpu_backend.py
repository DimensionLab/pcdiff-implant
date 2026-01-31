"""
Pure PyTorch CPU implementations of PVCNN operations.

These are fallback implementations for when CUDA is not available (e.g., macOS, CPU-only systems).
They are slower than the CUDA kernels but functionally equivalent.
"""
import torch
import torch.nn.functional as F


def ball_query(centers_coords, points_coords, radius, num_neighbors):
    """
    Ball query: find neighbors within radius.

    :param centers_coords: FloatTensor[B, 3, M] - coordinates of centers
    :param points_coords: FloatTensor[B, 3, N] - coordinates of points
    :param radius: float - radius of ball query
    :param num_neighbors: int - maximum number of neighbors
    :return: IntTensor[B, M, U] - neighbor indices
    """
    B, _, M = centers_coords.shape
    _, _, N = points_coords.shape
    device = centers_coords.device

    # Transpose for easier distance computation: [B, M, 3] and [B, N, 3]
    centers = centers_coords.transpose(1, 2)  # [B, M, 3]
    points = points_coords.transpose(1, 2)    # [B, N, 3]

    # Compute pairwise squared distances: [B, M, N]
    # ||c - p||^2 = ||c||^2 + ||p||^2 - 2 * c.p
    centers_sq = (centers ** 2).sum(dim=-1, keepdim=True)  # [B, M, 1]
    points_sq = (points ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # [B, 1, N]
    cross = torch.bmm(centers, points.transpose(1, 2))  # [B, M, N]
    dists_sq = centers_sq + points_sq - 2 * cross  # [B, M, N]

    # Find neighbors within radius
    radius_sq = radius * radius
    mask = dists_sq <= radius_sq  # [B, M, N]

    # For each center, get the first num_neighbors indices
    # Initialize with zeros (will point to first point as fallback)
    neighbor_indices = torch.zeros(B, M, num_neighbors, dtype=torch.int32, device=device)

    for b in range(B):
        for m in range(M):
            valid_indices = mask[b, m].nonzero(as_tuple=False).squeeze(-1)
            n_valid = min(len(valid_indices), num_neighbors)
            if n_valid > 0:
                # Sort by distance and take closest
                valid_dists = dists_sq[b, m, valid_indices]
                sorted_idx = valid_dists.argsort()[:n_valid]
                neighbor_indices[b, m, :n_valid] = valid_indices[sorted_idx].int()
                # Fill remaining with the last valid index
                if n_valid < num_neighbors:
                    neighbor_indices[b, m, n_valid:] = neighbor_indices[b, m, n_valid - 1]

    return neighbor_indices


def grouping_forward(features, indices):
    """
    Group features by indices.

    :param features: FloatTensor[B, C, N] - features of points
    :param indices: IntTensor[B, M, U] - neighbor indices
    :return: FloatTensor[B, C, M, U] - grouped features
    """
    B, C, N = features.shape
    _, M, U = indices.shape
    device = features.device

    # Expand indices for gather: [B, C, M, U]
    indices_expanded = indices.unsqueeze(1).expand(B, C, M, U).long()

    # Expand features: [B, C, N] -> [B, C, 1, N]
    features_expanded = features.unsqueeze(2).expand(B, C, M, N)

    # Gather along the N dimension
    # We need to flatten and gather
    grouped = torch.zeros(B, C, M, U, device=device, dtype=features.dtype)
    for b in range(B):
        for c in range(C):
            grouped[b, c] = features[b, c, indices[b].long()]

    return grouped


def grouping_backward(grad_output, indices, num_points):
    """
    Backward pass for grouping.

    :param grad_output: FloatTensor[B, C, M, U] - gradient of output
    :param indices: IntTensor[B, M, U] - neighbor indices
    :param num_points: int - number of points N
    :return: FloatTensor[B, C, N] - gradient of features
    """
    B, C, M, U = grad_output.shape
    device = grad_output.device

    grad_features = torch.zeros(B, C, num_points, device=device, dtype=grad_output.dtype)

    for b in range(B):
        for m in range(M):
            for u in range(U):
                idx = indices[b, m, u].long()
                grad_features[b, :, idx] += grad_output[b, :, m, u]

    return grad_features


def gather_features_forward(features, indices):
    """
    Gather features at indices.

    :param features: FloatTensor[B, C, N] - features
    :param indices: IntTensor[B, M] - indices to gather
    :return: FloatTensor[B, C, M] - gathered features
    """
    B, C, N = features.shape
    _, M = indices.shape

    # Use index_select via gather
    indices_expanded = indices.unsqueeze(1).expand(B, C, M).long()
    gathered = torch.gather(features, 2, indices_expanded)

    return gathered


def gather_features_backward(grad_output, indices, num_points):
    """
    Backward pass for gather.

    :param grad_output: FloatTensor[B, C, M] - gradient of output
    :param indices: IntTensor[B, M] - indices
    :param num_points: int - N
    :return: FloatTensor[B, C, N] - gradient of features
    """
    B, C, M = grad_output.shape
    device = grad_output.device

    grad_features = torch.zeros(B, C, num_points, device=device, dtype=grad_output.dtype)
    indices_expanded = indices.unsqueeze(1).expand(B, C, M).long()
    grad_features.scatter_add_(2, indices_expanded, grad_output)

    return grad_features


def furthest_point_sampling(coords, num_samples):
    """
    Iterative furthest point sampling.

    :param coords: FloatTensor[B, 3, N] - coordinates
    :param num_samples: int - number of samples M
    :return: IntTensor[B, M] - indices of sampled points
    """
    B, _, N = coords.shape
    device = coords.device

    # Transpose to [B, N, 3]
    coords_t = coords.transpose(1, 2)

    indices = torch.zeros(B, num_samples, dtype=torch.int32, device=device)

    for b in range(B):
        pts = coords_t[b]  # [N, 3]

        # Start with a random point
        farthest = torch.randint(0, N, (1,), device=device).item()
        indices[b, 0] = farthest

        # Distance from each point to the sampled set
        distances = torch.full((N,), float('inf'), device=device)

        for i in range(1, num_samples):
            # Update distances
            centroid = pts[farthest]  # [3]
            dist_to_centroid = ((pts - centroid) ** 2).sum(dim=-1)  # [N]
            distances = torch.minimum(distances, dist_to_centroid)

            # Select the farthest point
            farthest = distances.argmax().item()
            indices[b, i] = farthest

    return indices


def avg_voxelize_forward(features, coords, resolution):
    """
    Average voxelization.

    :param features: FloatTensor[B, C, N] - features
    :param coords: IntTensor[B, 3, N] - voxel coordinates
    :param resolution: int - voxel resolution R
    :return: tuple(out, indices, counts)
        - out: FloatTensor[B, C, R^3]
        - indices: IntTensor[B, N] - flattened voxel indices
        - counts: IntTensor[B, R^3] - count per voxel
    """
    B, C, N = features.shape
    R = resolution
    device = features.device

    # Flatten voxel coordinates to linear indices
    # coords: [B, 3, N] with values in [0, R-1]
    indices = coords[:, 0, :] * R * R + coords[:, 1, :] * R + coords[:, 2, :]  # [B, N]
    indices = indices.long()

    # Initialize output and counts
    out = torch.zeros(B, C, R * R * R, device=device, dtype=features.dtype)
    counts = torch.zeros(B, R * R * R, device=device, dtype=torch.int32)

    for b in range(B):
        # Scatter add features
        idx_expanded = indices[b].unsqueeze(0).expand(C, N)  # [C, N]
        out[b].scatter_add_(1, idx_expanded, features[b])

        # Count points per voxel
        ones = torch.ones(N, dtype=torch.int32, device=device)
        counts[b].scatter_add_(0, indices[b], ones)

    # Average (avoid division by zero)
    counts_expanded = counts.unsqueeze(1).expand(B, C, R * R * R).float()
    counts_expanded = torch.clamp(counts_expanded, min=1.0)
    out = out / counts_expanded

    return out, indices.int(), counts


def avg_voxelize_backward(grad_output, indices, counts):
    """
    Backward pass for average voxelization.

    :param grad_output: FloatTensor[B, C, R^3] - gradient
    :param indices: IntTensor[B, N] - voxel indices
    :param counts: IntTensor[B, R^3] - counts
    :return: FloatTensor[B, C, N] - gradient of features
    """
    B, C, _ = grad_output.shape
    N = indices.shape[1]
    device = grad_output.device

    # Scale gradient by 1/count for each voxel
    counts_f = counts.float().clamp(min=1.0)  # [B, R^3]
    scaled_grad = grad_output / counts_f.unsqueeze(1)  # [B, C, R^3]

    # Gather gradient for each point
    grad_features = torch.zeros(B, C, N, device=device, dtype=grad_output.dtype)
    for b in range(B):
        idx_expanded = indices[b].long().unsqueeze(0).expand(C, N)  # [C, N]
        grad_features[b] = torch.gather(scaled_grad[b], 1, idx_expanded)

    return grad_features


def trilinear_devoxelize_forward(resolution, is_training, coords, features):
    """
    Trilinear devoxelization (interpolation from voxel grid to points).

    :param resolution: int - voxel resolution R
    :param is_training: bool - training mode
    :param coords: FloatTensor[B, 3, N] - point coordinates (normalized to [0, R-1])
    :param features: FloatTensor[B, C, R^3] - voxel features (flattened)
    :return: tuple(outs, inds, wgts)
        - outs: FloatTensor[B, C, N] - interpolated features
        - inds: IntTensor[B, 8, N] - corner voxel indices
        - wgts: FloatTensor[B, 8, N] - interpolation weights
    """
    B, C, _ = features.shape
    _, _, N = coords.shape
    R = resolution
    device = coords.device

    # Reshape features to [B, C, R, R, R]
    features_grid = features.view(B, C, R, R, R)

    # Normalize coordinates to [-1, 1] for grid_sample
    # coords are in [0, R-1], need to map to [-1, 1]
    coords_normalized = (coords / (R - 1)) * 2 - 1  # [B, 3, N]

    # Reshape for grid_sample: [B, N, 1, 1, 3] (D, H, W order = z, y, x)
    # grid_sample expects (x, y, z) in last dim
    grid = coords_normalized.transpose(1, 2).unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]

    # Use grid_sample for trilinear interpolation
    # Input: [B, C, D, H, W], grid: [B, N, 1, 1, 3]
    # Output: [B, C, N, 1, 1]
    outs = F.grid_sample(features_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
    outs = outs.squeeze(-1).squeeze(-1)  # [B, C, N]

    # For backward, we need indices and weights of 8 corners
    # This is a simplified version - for training, grid_sample handles backward automatically
    inds = torch.zeros(B, 8, N, dtype=torch.int32, device=device)
    wgts = torch.zeros(B, 8, N, dtype=torch.float32, device=device)

    if is_training:
        # Compute corner indices and weights for backward pass
        coords_t = coords.transpose(1, 2)  # [B, N, 3]

        # Floor and ceil coordinates
        coords_floor = coords_t.floor().long().clamp(0, R - 1)  # [B, N, 3]
        coords_ceil = (coords_floor + 1).clamp(0, R - 1)  # [B, N, 3]

        # Fractional part for weights
        frac = coords_t - coords_floor.float()  # [B, N, 3]

        # 8 corners (x, y, z combinations of floor/ceil)
        # Index order: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        for i in range(8):
            x_idx = coords_ceil[:, :, 0] if (i & 4) else coords_floor[:, :, 0]
            y_idx = coords_ceil[:, :, 1] if (i & 2) else coords_floor[:, :, 1]
            z_idx = coords_ceil[:, :, 2] if (i & 1) else coords_floor[:, :, 2]

            # Flatten index
            flat_idx = x_idx * R * R + y_idx * R + z_idx  # [B, N]
            inds[:, i, :] = flat_idx.int()

            # Trilinear weight
            wx = frac[:, :, 0] if (i & 4) else (1 - frac[:, :, 0])
            wy = frac[:, :, 1] if (i & 2) else (1 - frac[:, :, 1])
            wz = frac[:, :, 2] if (i & 1) else (1 - frac[:, :, 2])
            wgts[:, i, :] = wx * wy * wz

    return outs, inds, wgts


def trilinear_devoxelize_backward(grad_output, inds, wgts, resolution):
    """
    Backward pass for trilinear devoxelization.

    :param grad_output: FloatTensor[B, C, N] - gradient of output
    :param inds: IntTensor[B, 8, N] - corner indices
    :param wgts: FloatTensor[B, 8, N] - interpolation weights
    :param resolution: int - R
    :return: FloatTensor[B, C, R^3] - gradient of voxel features
    """
    B, C, N = grad_output.shape
    R = resolution
    device = grad_output.device

    grad_features = torch.zeros(B, C, R * R * R, device=device, dtype=grad_output.dtype)

    for b in range(B):
        for corner in range(8):
            # Weight gradient and scatter to voxels
            weighted_grad = grad_output[b] * wgts[b, corner].unsqueeze(0)  # [C, N]
            idx = inds[b, corner].long().unsqueeze(0).expand(C, N)  # [C, N]
            grad_features[b].scatter_add_(1, idx, weighted_grad)

    return grad_features


def three_nearest_neighbors_interpolate_forward(points_coords, centers_coords, centers_features):
    """
    3-nearest neighbor interpolation.

    :param points_coords: FloatTensor[B, 3, N] - coordinates of query points
    :param centers_coords: FloatTensor[B, 3, M] - coordinates of centers
    :param centers_features: FloatTensor[B, C, M] - features of centers
    :return: tuple(points_features, indices, weights)
    """
    B, _, N = points_coords.shape
    _, C, M = centers_features.shape
    device = points_coords.device

    # Transpose for distance computation
    points = points_coords.transpose(1, 2)  # [B, N, 3]
    centers = centers_coords.transpose(1, 2)  # [B, M, 3]

    # Compute pairwise distances: [B, N, M]
    points_sq = (points ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    centers_sq = (centers ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # [B, 1, M]
    cross = torch.bmm(points, centers.transpose(1, 2))  # [B, N, M]
    dists = points_sq + centers_sq - 2 * cross  # [B, N, M]
    dists = dists.clamp(min=1e-10).sqrt()  # [B, N, M]

    # Find 3 nearest neighbors
    _, indices = dists.topk(3, dim=-1, largest=False)  # [B, N, 3]
    indices = indices.int()

    # Gather distances to 3 nearest
    knn_dists = torch.gather(dists, 2, indices.long())  # [B, N, 3]

    # Inverse distance weighting
    inv_dists = 1.0 / knn_dists.clamp(min=1e-10)  # [B, N, 3]
    weights = inv_dists / inv_dists.sum(dim=-1, keepdim=True)  # [B, N, 3]

    # Interpolate features
    points_features = torch.zeros(B, C, N, device=device, dtype=centers_features.dtype)
    for k in range(3):
        idx = indices[:, :, k].long()  # [B, N]
        idx_expanded = idx.unsqueeze(1).expand(B, C, N)  # [B, C, N]
        gathered = torch.gather(centers_features, 2, idx_expanded)  # [B, C, N]
        points_features += gathered * weights[:, :, k].unsqueeze(1)

    return points_features, indices.transpose(1, 2), weights.transpose(1, 2)


def three_nearest_neighbors_interpolate_backward(grad_output, indices, weights, num_centers):
    """
    Backward pass for 3-nearest neighbor interpolation.

    :param grad_output: FloatTensor[B, C, N]
    :param indices: IntTensor[B, 3, N]
    :param weights: FloatTensor[B, 3, N]
    :param num_centers: int - M
    :return: FloatTensor[B, C, M]
    """
    B, C, N = grad_output.shape
    device = grad_output.device

    grad_centers = torch.zeros(B, C, num_centers, device=device, dtype=grad_output.dtype)

    for k in range(3):
        idx = indices[:, k, :].long()  # [B, N]
        idx_expanded = idx.unsqueeze(1).expand(B, C, N)  # [B, C, N]
        weighted_grad = grad_output * weights[:, k, :].unsqueeze(1)  # [B, C, N]
        grad_centers.scatter_add_(2, idx_expanded, weighted_grad)

    return grad_centers


class CPUBackend:
    """Wrapper class to match the interface of the CUDA backend."""

    @staticmethod
    def ball_query(centers_coords, points_coords, radius, num_neighbors):
        return ball_query(centers_coords, points_coords, radius, num_neighbors)

    @staticmethod
    def grouping_forward(features, indices):
        return grouping_forward(features, indices)

    @staticmethod
    def grouping_backward(grad_output, indices, num_points):
        return grouping_backward(grad_output, indices, num_points)

    @staticmethod
    def gather_features_forward(features, indices):
        return gather_features_forward(features, indices)

    @staticmethod
    def gather_features_backward(grad_output, indices, num_points):
        return gather_features_backward(grad_output, indices, num_points)

    @staticmethod
    def furthest_point_sampling(coords, num_samples):
        return furthest_point_sampling(coords, num_samples)

    @staticmethod
    def avg_voxelize_forward(features, coords, resolution):
        return avg_voxelize_forward(features, coords, resolution)

    @staticmethod
    def avg_voxelize_backward(grad_output, indices, counts):
        return avg_voxelize_backward(grad_output, indices, counts)

    @staticmethod
    def trilinear_devoxelize_forward(resolution, is_training, coords, features):
        return trilinear_devoxelize_forward(resolution, is_training, coords, features)

    @staticmethod
    def trilinear_devoxelize_backward(grad_output, inds, wgts, resolution):
        return trilinear_devoxelize_backward(grad_output, inds, wgts, resolution)

    @staticmethod
    def three_nearest_neighbors_interpolate_forward(points_coords, centers_coords, centers_features):
        return three_nearest_neighbors_interpolate_forward(points_coords, centers_coords, centers_features)

    @staticmethod
    def three_nearest_neighbors_interpolate_backward(grad_output, indices, weights, num_centers):
        return three_nearest_neighbors_interpolate_backward(grad_output, indices, weights, num_centers)


# Singleton instance
_cpu_backend = CPUBackend()

__all__ = ['_cpu_backend']
