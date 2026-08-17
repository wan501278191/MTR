"""Test-Time Augmentation utilities for MTR inference.

Applies 4 geometric augmentations (identity, h-flip, v-flip, 180° rotation)
to each batch, runs the model, then inverse-transforms and clusters the
predictions back to ``num_modes`` trajectories.
"""

import numpy as np
import torch

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Geometric augmentation helpers (operate on coordinate tensors)
# ---------------------------------------------------------------------------

def _flip_x(trajs):
    """Flip along x-axis (negate x coordinate)."""
    out = trajs.clone()
    out[..., 0] = -out[..., 0]
    return out


def _flip_y(trajs):
    """Flip along y-axis (negate y coordinate)."""
    out = trajs.clone()
    out[..., 1] = -out[..., 1]
    return out


def _rotate_180(trajs):
    """Rotate 180 degrees (negate both coordinates)."""
    out = trajs.clone()
    out[..., 0] = -out[..., 0]
    out[..., 1] = -out[..., 1]
    return out


AUGMENTATIONS = [
    ('identity', lambda t: t),
    ('flip_x', _flip_x),
    ('flip_y', _flip_y),
    ('rotate_180', _rotate_180),
]


def inverse_transform_trajs(trajs, aug_name):
    """Apply the inverse of ``aug_name`` to predicted trajectories."""
    if aug_name == 'identity':
        return trajs
    if aug_name == 'flip_x':
        return _flip_x(trajs)
    if aug_name == 'flip_y':
        return _flip_y(trajs)
    if aug_name == 'rotate_180':
        return _rotate_180(trajs)
    raise ValueError(f'Unknown augmentation: {aug_name}')


def augment_input_dict(input_dict, aug_name):
    """Apply a geometric augmentation to the input dictionary coordinates.

    Only the coordinate arrays that participate in trajectory prediction are
    transformed: ``obj_trajs`` (x,y positions), ``obj_trajs_pos``,
    ``obj_trajs_last_pos``, ``map_polylines``, ``map_polylines_center``,
    and ``center_objects_world`` (position + heading).

    Velocities are left as-is; only positional x/y components are flipped
    consistently.  This is a best-effort TTA that preserves relative geometry.
    """
    if aug_name == 'identity':
        return input_dict

    def flip_xy(arr):
        arr = arr.clone()
        arr[..., 0] = -arr[..., 0]
        return arr

    def flip_yy(arr):
        arr = arr.clone()
        arr[..., 1] = -arr[..., 1]
        return arr

    def rot(arr):
        arr = arr.clone()
        arr[..., 0] = -arr[..., 0]
        arr[..., 1] = -arr[..., 1]
        return arr

    fn = {'flip_x': flip_xy, 'flip_y': flip_yy, 'rotate_180': rot}[aug_name]

    new_dict = {}
    for k, v in input_dict.items():
        new_dict[k] = v

    for key in ['obj_trajs', 'obj_trajs_pos', 'obj_trajs_last_pos',
                'map_polylines', 'map_polylines_center', 'obj_trajs_future_state',
                'center_gt_trajs']:
        if key in new_dict and torch.is_tensor(new_dict[key]) and new_dict[key].shape[-1] >= 2:
            new_dict[key] = fn(new_dict[key])

    # center_objects_world: (N, 10) [x, y, z, ... heading(angle), ...]
    if 'center_objects_world' in new_dict and torch.is_tensor(new_dict['center_objects_world']):
        cow = new_dict['center_objects_world'].clone()
        cow[:, 0] = -cow[:, 0] if aug_name in ('flip_x', 'rotate_180') else cow[:, 0]
        cow[:, 1] = -cow[:, 1] if aug_name in ('flip_y', 'rotate_180') else cow[:, 1]
        # heading angle: flip_x -> -theta, flip_y -> pi-theta, rotate_180 -> theta+pi
        heading = cow[:, 6]
        if aug_name == 'flip_x':
            heading = -heading
        elif aug_name == 'flip_y':
            heading = np.pi - heading
        elif aug_name == 'rotate_180':
            heading = heading + np.pi
        cow[:, 6] = heading
        new_dict['center_objects_world'] = cow

    return new_dict


# ---------------------------------------------------------------------------
# Clustering / merging
# ---------------------------------------------------------------------------

def cluster_and_merge(all_preds, num_modes=6):
    """Cluster a list of per-augmentation predictions into ``num_modes`` modes.

    Each prediction dict has:
        pred_trajs: (num_center_objects, M, T, F)  where F >= 2 (xy + optional GMM params)
        pred_scores: (num_center_objects, M)

    Clustering is performed on the xy component only, but the full-feature
    trajectory of the cluster representative is preserved.
    Returns merged dict with ``num_modes`` modes per center object.
    """
    if not _HAS_SKLEARN:
        return all_preds[0]

    pred_trajs = all_preds[0]['pred_trajs']  # (N, M, T, 2)
    num_center_objects = pred_trajs.shape[0]

    merged_trajs = []
    merged_scores = []
    for obj_idx in range(num_center_objects):
        # gather all modes across augmentations: (num_aug * M, T, 2)
        all_trajs = torch.cat([p['pred_trajs'][obj_idx] for p in all_preds], dim=0)
        all_scores = torch.cat([p['pred_scores'][obj_idx] for p in all_preds], dim=0)

        num_total = all_trajs.shape[0]
        if num_total <= num_modes:
            merged_trajs.append(all_trajs)
            merged_scores.append(all_scores)
            continue

        xy = all_trajs[..., 0:2].reshape(num_total, -1).cpu().numpy()
        kmeans = KMeans(n_clusters=num_modes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(xy)

        obj_trajs = []
        obj_scores = []
        for c in range(num_modes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            cluster_scores = all_scores[mask]
            best_idx = cluster_scores.argmax()
            obj_trajs.append(all_trajs[mask][best_idx])
            obj_scores.append(cluster_scores[best_idx])

        obj_trajs = torch.stack(obj_trajs, dim=0)
        obj_scores = torch.stack(obj_scores, dim=0)
        obj_scores = obj_scores / obj_scores.sum().clamp_min(1e-6)

        merged_trajs.append(obj_trajs)
        merged_scores.append(obj_scores)

    # pad to num_modes if some objects had fewer clusters
    final_trajs = []
    final_scores = []
    for t, s in zip(merged_trajs, merged_scores):
        if t.shape[0] < num_modes:
            pad_t = t.new_zeros(num_modes - t.shape[0], *t.shape[1:])
            pad_s = s.new_zeros(num_modes - s.shape[0])
            t = torch.cat([t, pad_t], dim=0)
            s = torch.cat([s, pad_s], dim=0)
        final_trajs.append(t)
        final_scores.append(s)

    return {
        'pred_trajs': torch.stack(final_trajs, dim=0),
        'pred_scores': torch.stack(final_scores, dim=0),
    }
