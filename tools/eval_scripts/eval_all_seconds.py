"""
赛事官方评估脚本 — 一次性评估 3s/5s/8s 全部指标并同步到 SwanLab。

用法:
    cd MTR/tools/eval_scripts
    python eval_all_seconds.py \
        --pred_file ../../output/waymo/mtr_voyah_data/eval_baseline/eval/epoch_30/default/result.pkl \
        --gt_file ../../../data/gt_data_val.pkl \
        --extra_tag eval_baseline
"""
import argparse
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from waymo_eval import waymo_evaluation

# SwanLab 需要在 MTR 包路径可用时才能 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mtr.utils.swanlab_logger import SwanLabWriter

# 抑制 TF GPU 占用（评估不需要 GPU）
all_gpus = tf.config.experimental.list_physical_devices('GPU')
if all_gpus:
    for gpu in all_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

TOP_K = 6
NUM_MODES_FOR_EVAL = 6
EVAL_SECONDS = [3, 5, 8]
OBJECT_TYPES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
METRICS = ['mAP', 'minADE', 'minFDE', 'MissRate']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3s/5s/8s metrics and sync to SwanLab')
    parser.add_argument('--pred_file', type=str, required=True, help='Path to result.pkl')
    parser.add_argument('--gt_file', type=str, required=True, help='Path to gt_data.pkl')
    parser.add_argument('--extra_tag', type=str, default='eval', help='SwanLab experiment name')
    parser.add_argument('--swanlab_project', type=str, default='MTR', help='SwanLab project name')
    return parser.parse_args()


def load_data(pred_file, gt_file):
    with open(pred_file, 'rb') as f:
        pred_data = pickle.load(f)
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    return pred_data, gt_data


def format_for_evaluation(pred_data, gt_data):
    eval_data = []
    for pred_scene in pred_data:
        formatted_scene = []
        scenario_id = pred_scene[0]['scenario_id']

        if scenario_id not in gt_data:
            print(f"Warning: scenario_id {scenario_id} not found in ground truth data.")
            continue

        gt_scene = gt_data[scenario_id]
        track_indices = np.atleast_1d(gt_scene['tracks_to_predict']).tolist()
        assert len(pred_scene) == len(track_indices), \
            f"pred_scene len={len(pred_scene)} != track_indices len={len(track_indices)}, scenario={scenario_id}"

        for i, obj_pred in enumerate(pred_scene):
            obj_id = obj_pred['object_id']
            if obj_id not in gt_scene['object_id']:
                print(f"Warning: object_id {obj_id} in scenario {scenario_id} not found in GT.")
                continue

            gt_idx = track_indices[i]
            pred_trajs = obj_pred['pred_trajs'][:, :80, :]
            gt_trajs = gt_scene['gt_trajs'][gt_idx][:91, :]
            scores = obj_pred['pred_scores']

            sorted_indices = np.argsort(-scores)
            topk_indices = sorted_indices[:TOP_K]
            topk_scores = scores[topk_indices]
            topk_trajs = pred_trajs[topk_indices, :, :]

            formatted_obj = {
                'scenario_id': scenario_id,
                'pred_trajs': topk_trajs,
                'pred_scores': topk_scores,
                'object_id': obj_id,
                'object_type': obj_pred['object_type'],
                'gt_trajs': gt_trajs
            }
            formatted_scene.append(formatted_obj)

        if formatted_scene:
            eval_data.append(formatted_scene)
    return eval_data


def run_eval_one_second(eval_data, eval_second):
    """Run Waymo evaluation for a single time window (3/5/8s)."""
    metric_results, result_format_str = waymo_evaluation(
        pred_dicts=eval_data,
        top_k=TOP_K,
        eval_second=eval_second,
        num_modes_for_eval=NUM_MODES_FOR_EVAL
    )
    return metric_results, result_format_str


def extract_metrics(metric_results, eval_second):
    """Extract 12 key metrics (3 types x 4 metrics) from raw result dict.

    Returns dict like:
        {
            'eval_3s.mAP_VEHICLE': 0.48,
            'eval_3s.minADE_VEHICLE': 0.85,
            ...
            'eval_3s.mAP_avg': 0.48,
            ...
        }
    """
    extracted = {}

    for obj_type in OBJECT_TYPES:
        for metric in METRICS:
            # Waymo eval uses keys like 'minADE - VEHICLE', 'mAP - PEDESTRIAN'
            key = f'{metric} - {obj_type}'
            if key in metric_results:
                val = float(metric_results[key])
            else:
                val = 0.0
            swanlab_key = f'eval_{eval_second}s.{metric}_{obj_type}'
            extracted[swanlab_key] = val

    # Average across 3 object types
    for metric in METRICS:
        vals = []
        for obj_type in OBJECT_TYPES:
            key = f'{metric} - {obj_type}'
            if key in metric_results:
                vals.append(float(metric_results[key]))
        avg = sum(vals) / len(vals) if vals else 0.0
        extracted[f'eval_{eval_second}s.{metric}_avg'] = avg

    return extracted


def main():
    args = parse_args()

    # Step 1: Load data
    print(f"Loading pred: {args.pred_file}")
    print(f"Loading GT: {args.gt_file}")
    pred_data, gt_data = load_data(args.pred_file, args.gt_file)
    print(f"Loaded {len(pred_data)} predicted scenes, {len(gt_data)} GT scenes")

    # Step 2: Format eval data (shared across all time windows)
    eval_data = format_for_evaluation(pred_data, gt_data)
    print(f"Formatted {len(eval_data)} scenes for evaluation")

    # Step 3: Initialize SwanLab
    swanlab_log = SwanLabWriter(
        project=args.swanlab_project,
        name=args.extra_tag + '_eval',
        log_dir=os.path.join(os.path.dirname(args.pred_file), 'swanlab_eval'),
        config={
            'pred_file': args.pred_file,
            'gt_file': args.gt_file,
            'top_k': TOP_K,
            'num_modes': NUM_MODES_FOR_EVAL,
        }
    )

    # Step 4: Evaluate 3s / 5s / 8s
    all_metrics = {}
    for eval_second in EVAL_SECONDS:
        print(f"\n{'='*60}")
        print(f"Evaluating {eval_second}s...")
        print(f"{'='*60}")

        metric_results, result_format_str = run_eval_one_second(eval_data, eval_second)

        # Print results
        print(f"\n--- {eval_second}s Results ---")
        print(result_format_str)

        # Extract and log key metrics
        extracted = extract_metrics(metric_results, eval_second)
        all_metrics.update(extracted)

        for key, val in sorted(extracted.items()):
            print(f"  {key}: {val:.4f}")

    # Step 5: Log all metrics to SwanLab
    print(f"\n{'='*60}")
    print("Syncing all metrics to SwanLab...")
    print(f"{'='*60}")

    for key, val in sorted(all_metrics.items()):
        swanlab_log.add_scalar(key, val, step=0)

    # Step 6: Print summary table
    print(f"\n{'='*60}")
    print("Summary: 3s / 5s / 8s")
    print(f"{'='*60}")
    print(f"{'Metric':<35s} {'3s':>8s} {'5s':>8s} {'8s':>8s}")
    print("-" * 62)

    for obj_type in OBJECT_TYPES + ['avg']:
        for metric in METRICS:
            row_name = f"{metric}_{obj_type}"
            vals = []
            for es in EVAL_SECONDS:
                key = f'eval_{es}s.{metric}_{obj_type}'
                vals.append(f"{all_metrics.get(key, 0):.4f}")
            print(f"{row_name:<35s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

    swanlab_log.close()
    print(f"\nDone. SwanLab experiment: {args.extra_tag}_eval")


if __name__ == '__main__':
    main()
