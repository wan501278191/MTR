# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time

import numpy as np
import torch
import tqdm

from mtr.utils import common_utils

import tta_utils


def _tta_inference(model, dataset, batch_dict, output_path, num_modes=6):
    """Run TTA: 4 augmentations -> inverse transform -> cluster merge."""
    import copy
    import torch

    all_preds = []
    for aug_name, _ in tta_utils.AUGMENTATIONS:
        aug_batch = copy.copy(batch_dict)
        aug_batch['input_dict'] = tta_utils.augment_input_dict(batch_dict['input_dict'], aug_name)
        pred = model(aug_batch)
        if 'pred_trajs' not in pred:
            continue
        # inverse-transform predicted trajectories back to original frame
        pred_trajs = pred['pred_trajs']  # (N, M, T, 7) in center-object frame
        pred_trajs_xy = pred_trajs[:, :, :, 0:2]
        pred_trajs_xy = tta_utils.inverse_transform_trajs(pred_trajs_xy, aug_name)
        pred_trajs = pred_trajs.clone()
        pred_trajs[:, :, :, 0:2] = pred_trajs_xy
        pred['pred_trajs'] = pred_trajs
        all_preds.append(pred)

    if len(all_preds) <= 1:
        batch_pred_dicts = model(batch_dict)
        return dataset.generate_prediction_dicts(batch_pred_dicts, output_path=output_path)

    # collect per-object predictions
    merged = tta_utils.cluster_and_merge(all_preds, num_modes=num_modes)
    # build a fake batch_dict with merged predictions for generate_prediction_dicts
    merged_batch = {**all_preds[0]}
    merged_batch['pred_trajs'] = merged['pred_trajs']
    merged_batch['pred_scores'] = merged['pred_scores']
    return dataset.generate_prediction_dicts(merged_batch, output_path=output_path)


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    use_tta = cfg.get('TTA', {}).get('ENABLED', False)
    tta_num_modes = cfg.get('TTA', {}).get('NUM_MODES', cfg.MODEL.MOTION_DECODER.NUM_MOTION_MODES)

    pred_dicts = []
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            if use_tta:
                final_pred_dicts = _tta_inference(
                    model, dataset, batch_dict, final_output_dir if save_to_file else None,
                    num_modes=tta_num_modes,
                )
            else:
                batch_pred_dicts = model(batch_dict)
                final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        if pred_dicts is not None:
            logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
