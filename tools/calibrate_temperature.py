"""Post-training temperature scaling for MTR confidence calibration.

Usage:
    python tools/calibrate_temperature.py \
        --cfg_file tools/cfgs/waymo/mtr+100_percent_data.yaml \
        --ckpt output/.../best_model.pth \
        --calib_split test
"""

import _init_path
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from mtr.config import cfg, cfg_from_yaml_file, cfg_from_list, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Temperature calibration')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--calib_split', type=str, default='test',
                        help='which split to use for calibration')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--extra_tag', type=str, default='calibration')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def fit_temperature(model, dataloader, device, max_iter=50, lr=0.01):
    """Optimise a single temperature scalar on the validation set.

    Collects raw logits and GT nearest-mode indices, then runs LBFGS to
    minimise cross-entropy with temperature scaling.
    """
    model.eval()
    all_logits = []
    all_targets = []

    print('Collecting logits for calibration...')
    for batch_dict in tqdm.tqdm(dataloader):
        with torch.no_grad():
            model(batch_dict)
            # access decoder internals
            decoder = model.motion_decoder
            pred_list = decoder.forward_ret_dict['pred_list']
            pred_scores = pred_list[-1][0]  # (N, num_query) raw logits

            center_gt_trajs = decoder.forward_ret_dict['center_gt_trajs'].to(device)
            center_gt_trajs_mask = decoder.forward_ret_dict['center_gt_trajs_mask'].to(device)
            intention_points = decoder.forward_ret_dict['intention_points']  # (N, num_query, 2)

            num_center_objects = center_gt_trajs.shape[0]
            center_gt_final_valid_idx = decoder.forward_ret_dict['center_gt_final_valid_idx'].long()
            center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]

            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)
            target = dist.argmin(dim=-1)

            all_logits.append(pred_scores.cpu())
            all_targets.append(target.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperature
        loss = F.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_temp = temperature.item()
    print(f'Calibrated temperature: {final_temp:.4f}')
    return final_temp


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger(save_dir=Path('output') / 'calibration', rank=0)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        dist=False, workers=args.workers, logger=logger, training=False
    )

    model = model_utils.MotionTransformer(config=cfg.MODEL)
    model.load_params_from_file(args.ckpt, logger=logger, to_cpu=False)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    temp = fit_temperature(model, test_loader, model.motion_decoder.device,
                           max_iter=args.max_iter, lr=args.lr)

    # save temperature into the checkpoint
    model.motion_decoder.temperature.data.fill_(temp)
    save_path = Path(args.ckpt).parent / (Path(args.ckpt).stem + '_calibrated.pth')
    from train_utils.train_utils import checkpoint_state, save_checkpoint
    save_checkpoint(checkpoint_state(model, epoch=-1, it=0), filename=str(save_path))
    print(f'Saved calibrated checkpoint to {save_path}')


if __name__ == '__main__':
    main()
