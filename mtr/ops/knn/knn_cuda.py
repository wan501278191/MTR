"""
Pure PyTorch fallback for knn_cuda (CPU-compatible).
Provides the same interface as the CUDA extension but uses torch native ops.
"""
import torch


def knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k):
    """KNN within batches - pure PyTorch fallback.
    
    Args:
        xyz: (n, 3) float - source points
        query_xyz: (m, 3) float - query points
        batch_idxs: (n,) int - batch index for each source point
        query_batch_offsets: (B+1,) int - offsets for query batches
        idx: (n, k) int - output indices (modified in-place)
        n, m, k: sizes
    """
    B = query_batch_offsets.shape[0] - 1
    for b in range(B):
        # Get points in this batch
        src_mask = batch_idxs == b
        q_start = query_batch_offsets[b].item()
        q_end = query_batch_offsets[b + 1].item()

        src_pts = xyz[src_mask]  # (nb, 3)
        q_pts = query_xyz[q_start:q_end]  # (mb, 3)

        # Compute pairwise distances: (nb, mb)
        dist = torch.cdist(src_pts.unsqueeze(0), q_pts.unsqueeze(0)).squeeze(0)

        # Handle case where k > number of query points in batch
        actual_k = min(k, q_pts.shape[0])
        if actual_k > 0:
            _, local_indices = torch.topk(dist, actual_k, dim=1, largest=False)
            # Convert local indices to global indices
            global_indices = local_indices + q_start
            # Fill output
            src_indices = torch.where(src_mask)[0]
            idx[src_indices, :actual_k] = global_indices.to(idx.dtype)
            if actual_k < k:
                idx[src_indices, actual_k:] = -1


def knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k):
    """KNN with m*log(k) complexity - same as knn_batch for CPU fallback."""
    knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)
