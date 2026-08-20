import argparse
import pickle
import numpy as np
from waymo_eval import waymo_evaluation

def load_data(pred_file, gt_file):
    with open(pred_file, 'rb') as f:
        pred_data = pickle.load(f)
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    return pred_data, gt_data

def format_for_evaluation(pred_data, gt_data, top_k=6):
    eval_data = []
    for pred_scene in pred_data:
        formatted_scene = []
        scenario_id = str(pred_scene[0]['scenario_id'])

        if scenario_id not in gt_data:
            # 尝试去除 numpy/tensor 包装
            scenario_id_clean = scenario_id.strip("b'").strip("'")
            if scenario_id_clean in gt_data:
                scenario_id = scenario_id_clean
            else:
                print(f"Warning: scenario_id {scenario_id} not found in ground truth data.")
                continue

        gt_scene = gt_data[scenario_id]
        track_indices = np.atleast_1d(gt_scene['tracks_to_predict']).tolist()
        assert len(pred_scene) == len(track_indices), \
            f"预测数 {len(pred_scene)} != GT数 {len(track_indices)} (scenario={scenario_id})"
        i = 0
        for obj_pred in pred_scene:
            obj_id = int(obj_pred['object_id']) if not isinstance(obj_pred['object_id'], int) else obj_pred['object_id']
            gt_obj_ids = gt_scene['object_id']
            if obj_id not in gt_obj_ids:
                print(f"Warning: object_id {obj_id} in scenario {scenario_id} not found in ground truth data.")
                continue

            gt_idx = track_indices[i]
            pred_trajs = obj_pred['pred_trajs'][:, :80, :]
            gt_trajs = gt_scene['gt_trajs'][gt_idx][:91, :]
            scores = obj_pred['pred_scores']
            sorted_indices = np.argsort(-scores)
            topk_indices = sorted_indices[:top_k]
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
            i += 1

        if formatted_scene:
            eval_data.append(formatted_scene)
    return eval_data

def evaluate(pred_file, gt_file, top_k=6, eval_second=8, num_modes_for_eval=6):
    pred_data, gt_data = load_data(pred_file, gt_file)
    eval_data = format_for_evaluation(pred_data, gt_data, top_k)
    metric_results, result_format_str = waymo_evaluation(
        pred_dicts=eval_data,
        top_k=top_k,
        eval_second=eval_second,
        num_modes_for_eval=num_modes_for_eval
    )
    print(f"========== Evaluation @ {eval_second}s ==========")
    print(result_format_str)
    for key, value in metric_results.items():
        print(f"{key}: {value}")
    return metric_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估预测结果')
    parser.add_argument('--pred_file', type=str, required=True, help='result.pkl 路径')
    parser.add_argument('--gt_file', type=str, required=True, help='gt_data.pkl 路径')
    parser.add_argument('--eval_second', type=int, default=8, choices=[3, 5, 8], help='评估时间点')
    parser.add_argument('--top_k', type=int, default=6, help='Top-K')
    parser.add_argument('--num_modes', type=int, default=6, help='模态数')
    args = parser.parse_args()
    evaluate(args.pred_file, args.gt_file, args.top_k, args.eval_second, args.num_modes)
