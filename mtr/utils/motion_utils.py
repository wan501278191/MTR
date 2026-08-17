# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import math
import torch


def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1) # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs


def get_ade_of_waymo(pred_trajs, gt_trajs, gt_valid_mask, calculate_steps=(5, 9, 15)) -> float:
    """Compute Average Displacement Error.

    Args:
        pred_trajs: (batch_size, num_modes, pred_len, 2)
        gt_trajs: (batch_size, pred_len, 2)
        gt_valid_mask: (batch_size, pred_len)
    Returns:
        ade: Average Displacement Error

    """
    # assert pred_trajs.shape[2] in [1, 16, 80]
    if pred_trajs.shape[2] == 80:
        pred_trajs = pred_trajs[:, :, 4::5]
        gt_trajs = gt_trajs[:, 4::5]
        gt_valid_mask = gt_valid_mask[:, 4::5]

    ade = 0
    for cur_step in calculate_steps:
        dist_error = (pred_trajs[:, :, :cur_step+1, :] - gt_trajs[:, None, :cur_step+1, :]).norm(dim=-1)  # (batch_size, num_modes, pred_len)
        dist_error = (dist_error * gt_valid_mask[:, None, :cur_step+1].float()).sum(dim=-1) / torch.clamp_min(gt_valid_mask[:, :cur_step+1].sum(dim=-1)[:, None], min=1.0)  # (batch_size, num_modes)
        cur_ade = dist_error.min(dim=-1)[0].mean().item()

        ade += cur_ade

    ade = ade / len(calculate_steps)
    return ade


def get_ade_of_each_category(pred_trajs, gt_trajs, gt_trajs_mask, object_types, valid_type_list, post_tag='', pre_tag=''):
    """
    Args:
        pred_trajs (num_center_objects, num_modes, num_timestamps, 2): 
        gt_trajs (num_center_objects, num_timestamps, 2): 
        gt_trajs_mask (num_center_objects, num_timestamps): 
        object_types (num_center_objects): 

    Returns:
        
    """
    ret_dict = {}
    
    for cur_type in valid_type_list:
        type_mask = (object_types == cur_type)
        ret_dict[f'{pre_tag}ade_{cur_type}{post_tag}'] = -0.0
        if type_mask.sum() == 0:
            continue

        # calculate evaluataion metric
        ade = get_ade_of_waymo(
            pred_trajs=pred_trajs[type_mask, :, :, 0:2].detach(),
            gt_trajs=gt_trajs[type_mask], gt_valid_mask=gt_trajs_mask[type_mask]
        )
        ret_dict[f'{pre_tag}ade_{cur_type}{post_tag}'] = ade
    return ret_dict

def kinematic_filter(pred_trajs, pred_scores, frequency_hz=10.0,
                     max_speed=30.0, max_accel=8.0, max_steer_deg=35.0,
                     penalize_only=True):
    """Filter or penalise physically infeasible trajectories.

    Args:
        pred_trajs (torch.Tensor): (B, M, T, 2) predicted trajectories
        pred_scores (torch.Tensor): (B, M) confidence scores (already softmaxed)
        frequency_hz: trajectory sampling frequency (default 10 Hz)
        max_speed: maximum feasible speed (m/s)
        max_accel: maximum feasible acceleration (m/s^2)
        max_steer_deg: maximum heading change per step (degrees)
        penalize_only: if True, downweight infeasible modes instead of removing them

    Returns:
        filtered trajs (B, M, T, 2) and scores (B, M)
    """
    dt = 1.0 / frequency_hz
    max_steer = torch.deg2rad(torch.tensor(max_steer_deg, device=pred_trajs.device, dtype=pred_trajs.dtype))

    # velocities (B, M, T-1, 2)
    vel = pred_trajs[:, :, 1:, 0:2] - pred_trajs[:, :, :-1, 0:2]  # displacement per step
    speeds = vel.norm(dim=-1) / dt  # (B, M, T-1)

    headings = torch.atan2(vel[..., 1], vel[..., 0])  # (B, M, T-1)
    heading_diff = torch.abs(torch.diff(headings, dim=-1))  # (B, M, T-2)
    # wrap to [0, pi]
    heading_diff = torch.remainder(heading_diff, 2 * math.pi)
    heading_diff = torch.min(heading_diff, 2 * math.pi - heading_diff)

    speed_ok = speeds.max(dim=-1)[0] < max_speed  # (B, M)
    accel = torch.abs(torch.diff(speeds, dim=-1)) / dt  # (B, M, T-2)
    accel_ok = accel.max(dim=-1)[0] < max_accel
    steer_ok = heading_diff.max(dim=-1)[0] < max_steer

    feasible = speed_ok & accel_ok & steer_ok  # (B, M)

    if penalize_only:
        # downweight infeasible modes by 0.5
        penalty = torch.where(feasible, torch.ones_like(pred_scores), torch.full_like(pred_scores, 0.5))
        pred_scores = pred_scores * penalty
        pred_scores = pred_scores / pred_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    else:
        # hard filter: zero out infeasible, keep at least the best feasible mode
        pred_scores = pred_scores * feasible.float()
        max_scores, _ = pred_scores.max(dim=-1, keepdim=True)
        pred_scores = torch.where(max_scores > 0, pred_scores, torch.ones_like(pred_scores))

    return pred_trajs, pred_scores
