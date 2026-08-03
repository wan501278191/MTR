import pickle
import tensorflow as tf
import numpy as np
from waymo_eval import waymo_evaluation

# 直接定义预测和真值数据文件路径
PRED_FILE = './PATH/TO/RESULT/result.pkl'  # 替换为参赛者的预测文件路径
GT_FILE = './PATH/TO/GT/gt_data.pkl'  # 替换为真值数据文件路径
TOP_K = 6 # 规划改成1
EVAL_SECOND = 8 # 3/5/8s
NUM_MODES_FOR_EVAL = 6 # 规划改成1

def load_data(pred_file, gt_file):
    with open(pred_file, 'rb') as f:
        pred_data = pickle.load(f)
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    return pred_data, gt_data

def format_for_evaluation(pred_data, gt_data):
    """
    将预测数据与真实数据进行匹配，生成评估所需的 `eval_data` 格式。
    """
    eval_data = []
    for pred_scene in pred_data:
        formatted_scene = []
        scenario_id = pred_scene[0]['scenario_id']
        
        if scenario_id not in gt_data:
            print(f"Warning: scenario_id {scenario_id} not found in ground truth data.")
            continue
        
        gt_scene = gt_data[scenario_id]
        track_indices = np.atleast_1d(gt_scene['tracks_to_predict']).tolist()
        assert len(pred_scene) == len(track_indices)
        i = 0   # 注意：顺序索引track_to_predict，因此你的预测轨迹也需要顺序排列
        for obj_pred in pred_scene: 
            obj_id = obj_pred['object_id']
            if obj_id not in gt_scene['object_id']:
                print(f"Warning: object_id {obj_id} in scenario {scenario_id} not found in ground truth data.")
                continue

            gt_idx = track_indices[i]
            # 获取预测和真值轨迹，保证轨迹长度符合要求
            pred_trajs = obj_pred['pred_trajs'][:, :80, :]  # 预测轨迹长度限制为 80
            gt_trajs = gt_scene['gt_trajs'][gt_idx][:91, :]  # 真值轨迹长度限制为 91
            scores = obj_pred['pred_scores']  # shape: [M]
            sorted_indices = np.argsort(-scores)
            # 取前 TOP_K
            topk_indices = sorted_indices[:TOP_K]
            topk_scores = scores[topk_indices]
            topk_trajs = pred_trajs[topk_indices, :, :]

            # 创建格式化数据项
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

def evaluate():
    # Step 1: Load Prediction and Ground Truth Data
    pred_data, gt_data = load_data(PRED_FILE, GT_FILE)
    
    # Step 2: Format Prediction Data for Evaluation
    eval_data = format_for_evaluation(pred_data, gt_data)
    
    # Step 3: Perform Waymo Evaluation
    metric_results, result_format_str = waymo_evaluation(
        pred_dicts=eval_data,
        top_k=TOP_K,
        eval_second=EVAL_SECOND,
        num_modes_for_eval=NUM_MODES_FOR_EVAL
    )
    
    # Step 4: Output Results
    print("Evaluation Metrics:")
    print(result_format_str)
    for key, value in metric_results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    evaluate()
