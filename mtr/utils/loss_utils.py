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

class AnnealingScheduler:
    """Exponential temperature annealing for aWTA loss.

    High temperature at start -> all modes participate (prevent mode collapse).
    Low temperature at end -> concentrate on best mode (precision convergence).
    """

    def __init__(self, start_temp=10.0, end_temp=0.1, total_epochs=30):
        self.start_temp = float(start_temp)
        self.end_temp = float(end_temp)
        self.total_epochs = int(total_epochs)

    def get_temperature(self, epoch):
        ratio = min(float(epoch) / max(self.total_epochs, 1), 1.0)
        return self.start_temp * (self.end_temp / self.start_temp) ** ratio


def awta_nll_loss_gmm(pred_scores, pred_trajs, gt_trajs, gt_valid_mask,
                      temperature, pre_nearest_mode_idxs=None,
                      timestamp_loss_weight=None, use_square_gmm=False,
                      log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """Annealed Winner-Takes-All GMM NLL loss.

    Unlike ``nll_loss_gmm_direct`` which only backprops through the nearest
    mode, this computes the GMM NLL for *every* mode and produces a softmax
    (over modes) weighted combination using ``-nll / temperature`` as logits.

    As ``temperature -> 0`` the weighting concentrates on the best mode,
    recovering the standard WTA behaviour.
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size, num_modes = pred_scores.shape[0], pred_trajs.shape[1]

    # distance of every mode to GT (used for nearest-mode bookkeeping only)
    distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)  # (B, M, T)
    distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)  # (B, M)

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        nearest_mode_idxs = distance.argmin(dim=-1)

    # broadcast GT to all modes: (B, M, T, 2)
    gt_expanded = gt_trajs[:, None, :, :].expand(-1, num_modes, -1, -1)
    res_trajs = gt_expanded - pred_trajs[:, :, :, 0:2]  # (B, M, T, 2)
    dx = res_trajs[..., 0]
    dy = res_trajs[..., 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(pred_trajs[:, :, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(pred_trajs[:, :, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(pred_trajs[:, :, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)
        rho = torch.clip(pred_trajs[:, :, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask_f = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask_f = gt_valid_mask_f * timestamp_loss_weight[None, :]

    # per-mode, per-timestamp NLL  (B, M, T)
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)
    reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
        (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (std1 * std2)
    )
    nll_per_mode = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask_f[:, None, :]).sum(dim=-1)  # (B, M)

    # annealed softmax weights over modes
    weights = torch.softmax(-nll_per_mode / max(temperature, 1e-6), dim=1)  # (B, M)

    reg_loss = (weights * nll_per_mode).sum(dim=1)  # (B,)

    return reg_loss, nearest_mode_idxs
