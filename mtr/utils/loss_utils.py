# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch 
import torch.nn.functional as F


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

def laplace_nll_loss(pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                     timestamp_loss_weight=None, scale_min=0.01):
    """
    Laplace NLL loss for motion prediction.
    More robust to outliers than Gaussian NLL.

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 4): [mu_x, mu_y, log_scale_x, log_scale_y]
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        pre_nearest_mode_idxs (batch_size,): if provided, use it; else argmin
        timestamp_loss_weight (num_timestamps,):
    Returns:
        loss (batch_size,): per-sample Laplace NLL
        nearest_mode_idxs (batch_size,):
    """
    batch_size, num_modes = pred_trajs.shape[0], pred_trajs.shape[1]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
        nearest_mode_idxs = distance.argmin(dim=-1)

    nearest_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)
    nearest_trajs = pred_trajs[nearest_bs_idxs, nearest_mode_idxs]  # (B, T, 4)

    mu = nearest_trajs[:, :, 0:2]  # (B, T, 2)
    log_scale = nearest_trajs[:, :, 2:4]  # (B, T, 2)
    scale = torch.clamp(torch.exp(log_scale), min=scale_min)

    res = gt_trajs - mu  # (B, T, 2)
    # Laplace NLL: log(2*scale) + |res| / scale
    nll = torch.log(2 * scale) + torch.abs(res) / scale  # (B, T, 2)

    gt_valid = gt_valid_mask.type_as(pred_trajs)
    if timestamp_loss_weight is not None:
        gt_valid = gt_valid * timestamp_loss_weight[None, :]

    loss = (nll.sum(dim=-1) * gt_valid).sum(dim=-1)  # (B,)
    return loss, nearest_mode_idxs


def mixture_nll_loss(pred_scores, pred_trajs, gt_trajs, gt_valid_mask,
                     pre_nearest_mode_idxs=None, timestamp_loss_weight=None,
                     loss_type='gmm', use_square_gmm=False,
                     log_std_range=(-1.609, 5.0), rho_limit=0.5, scale_min=0.01):
    """
    Mixture NLL loss with soft mode selection via logsumexp.
    Borrowed from QCNet's MixtureNLLLoss — all modes receive gradient signal.

    Args:
        pred_scores (batch_size, num_modes): raw logits
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 4):
            GMM: [mu_x, mu_y, log_std_x, log_std_y, rho]
            Laplace: [mu_x, mu_y, log_scale_x, log_scale_y]
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        loss_type: 'gmm' or 'laplace'
    Returns:
        loss (batch_size,): mixture NLL (soft mode selection)
        nearest_mode_idxs (batch_size,): hard mode for velocity loss
    """
    batch_size, num_modes = pred_trajs.shape[0], pred_trajs.shape[1]
    num_timestamps = pred_trajs.shape[2]

    # Compute per-mode NLL (no pre-selection)
    if loss_type == 'gmm':
        nll_per_mode = _gmm_nll_per_mode(
            pred_trajs, gt_trajs, gt_valid_mask, timestamp_loss_weight,
            use_square_gmm, log_std_range, rho_limit
        )  # (B, num_modes)
    else:  # laplace
        nll_per_mode = _laplace_nll_per_mode(
            pred_trajs, gt_trajs, gt_valid_mask, timestamp_loss_weight, scale_min
        )  # (B, num_modes)

    # Soft mode selection: -logsumexp(log_pi - nll)
    log_pi = F.log_softmax(pred_scores, dim=-1)  # (B, num_modes)
    loss = -torch.logsumexp(log_pi - nll_per_mode, dim=-1)  # (B,)

    # Hard mode for velocity loss (still argmin for stability)
    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        nearest_mode_idxs = nll_per_mode.argmin(dim=-1)

    return loss, nearest_mode_idxs


def _gmm_nll_per_mode(pred_trajs, gt_trajs, gt_valid_mask, timestamp_loss_weight=None,
                      use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """Compute GMM NLL for all modes (not just nearest). Returns (B, num_modes)."""
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size, num_modes, num_timestamps, _ = pred_trajs.shape

    mu = pred_trajs[:, :, :, 0:2]  # (B, M, T, 2)
    res = gt_trajs[:, None, :, :] - mu  # (B, M, T, 2)
    dx, dy = res[..., 0], res[..., 1]

    if use_square_gmm:
        log_std = torch.clamp(pred_trajs[:, :, :, 2], min=log_std_range[0], max=log_std_range[1])
        std = torch.exp(log_std)
        rho = torch.zeros_like(log_std)
    else:
        log_std1 = torch.clamp(pred_trajs[:, :, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clamp(pred_trajs[:, :, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)
        rho = torch.clamp(pred_trajs[:, :, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid = gt_valid_mask.type_as(pred_trajs)
    if timestamp_loss_weight is not None:
        gt_valid = gt_valid * timestamp_loss_weight[None, :]

    if use_square_gmm:
        log_coeff = 2 * log_std
        exp_term = (dx ** 2 + dy ** 2) / (2 * std ** 2)
    else:
        log_coeff = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)
        exp_term = (0.5 / (1 - rho ** 2)) * (
            (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (std1 * std2)
        )

    nll = log_coeff + exp_term  # (B, M, T)
    nll = (nll * gt_valid[:, None, :]).sum(dim=-1)  # (B, M)
    return nll


def _laplace_nll_per_mode(pred_trajs, gt_trajs, gt_valid_mask, timestamp_loss_weight=None, scale_min=0.01):
    """Compute Laplace NLL for all modes. Returns (B, num_modes)."""
    assert pred_trajs.shape[-1] == 4

    mu = pred_trajs[:, :, :, 0:2]  # (B, M, T, 2)
    log_scale = pred_trajs[:, :, :, 2:4]  # (B, M, T, 2)
    scale = torch.clamp(torch.exp(log_scale), min=scale_min)

    res = gt_trajs[:, None, :, :] - mu  # (B, M, T, 2)
    nll = torch.log(2 * scale) + torch.abs(res) / scale  # (B, M, T, 2)
    nll = nll.sum(dim=-1)  # (B, M, T) sum over xy

    gt_valid = gt_valid_mask.type_as(pred_trajs)
    if timestamp_loss_weight is not None:
        gt_valid = gt_valid * timestamp_loss_weight[None, :]

    nll = (nll * gt_valid[:, None, :]).sum(dim=-1)  # (B, M)
    return nll
