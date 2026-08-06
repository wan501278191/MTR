# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch 


def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1) 
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) 

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs

def soft_target_cls_loss(pred_scores, dist_to_intention, topk=5, sigma=1.0, label_smoothing=0.1):
    """P0-1 + P0-3: Soft-target cross-entropy with label smoothing for intention classification.

    Replaces hard argmin assignment with a soft target distribution over the top-k
    nearest intention points, weighted by inverse distance. Provides gradient signal
    to multiple queries near the GT goal, mitigating boundary-intent undersupervision.

    Args:
        pred_scores (N, num_query): logits from motion_cls_head
        dist_to_intention (N, num_query): L2 distance from GT goal to each intention point
        topk: number of nearest intention points to assign soft targets
        sigma: temperature for distance weighting (larger = softer)
        label_smoothing: label smoothing factor (0 = no smoothing)

    Returns:
        loss (N,): per-sample soft-target cross-entropy
    """
    num_center_objects, num_query = dist_to_intention.shape
    topk = min(topk, num_query)
    topk_dist, topk_idx = dist_to_intention.topk(k=topk, dim=-1, largest=False)  # (N, topk)
    weights = torch.softmax(-topk_dist / (sigma + 1e-6), dim=-1)  # (N, topk)
    soft_target = pred_scores.new_zeros(num_center_objects, num_query)
    soft_target.scatter_(1, topk_idx, weights)
    if label_smoothing > 0:
        soft_target = soft_target * (1 - label_smoothing) + label_smoothing / num_query
    loss = torch.sum(-soft_target * torch.nn.functional.log_softmax(pred_scores, dim=-1), dim=-1)  # (N,)
    return loss


def compute_maneuver_label(gt_trajs, gt_final_valid_idx, num_classes=6, stationary_thresh=2.0):
    """P0-2: Compute maneuver class labels from GT trajectory endpoints.

    Classes:
        0: Stationary (total displacement < stationary_thresh)
        1..(num_classes-1): Angular sectors of endpoint direction

    Args:
        gt_trajs (N, T, 4): [x, y, vx, vy] in center object coordinate
        gt_final_valid_idx (N,): index of last valid future frame
        num_classes: total number of maneuver classes (>=2)
        stationary_thresh: displacement threshold for stationary class

    Returns:
        labels (N,): maneuver class indices
    """
    import math
    num_center_objects = gt_trajs.shape[0]
    idx = torch.arange(num_center_objects, device=gt_trajs.device)
    safe_idx = gt_final_valid_idx.to(gt_trajs.device).clamp(0, gt_trajs.shape[1] - 1)
    endpoints = gt_trajs[idx, safe_idx, 0:2]  # (N, 2)
    dx = endpoints[:, 0]
    dy = endpoints[:, 1]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    angle = torch.atan2(dy, dx)  # [-pi, pi]

    labels = torch.zeros(num_center_objects, dtype=torch.long, device=gt_trajs.device)
    stationary = dist < stationary_thresh
    labels[stationary] = 0
    non_stationary = ~stationary
    if num_classes > 1:
        num_angle_bins = num_classes - 1
        bin_width = 2 * math.pi / num_angle_bins
        angle_bins = torch.floor((angle + math.pi) / bin_width).long().clamp(0, num_angle_bins - 1) + 1
        labels[non_stationary] = angle_bins[non_stationary]
    return labels
