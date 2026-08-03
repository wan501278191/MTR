import os
import pickle

def create_gt_data_from_processed_files(processed_dir, output_file, prefix='enc_'):
    """
    从处理后的 Waymo 场景文件中提取 ground truth 轨迹信息，并保存至 gt_data.pkl
    Args:
        processed_dir (str): 已处理文件的目录，包含 `sample_{scenario_id}.pkl` 文件。
        output_file (str): 保存 `gt_data.pkl` 的路径。
    """
    gt_data = {}

    # 获取所有的处理后文件，以'sample_'开头并以'.pkl'结尾
    scene_files = [f for f in os.listdir(processed_dir) if f.startswith(prefix) and f.endswith('.pkl')]

    for scene_file in scene_files:
        file_path = os.path.join(processed_dir, scene_file)
        with open(file_path, 'rb') as f:
            scene_data = pickle.load(f)
        
        scenario_id = scene_data['scenario_id']
        
        # 提取真实轨迹信息
        gt_trajs = scene_data['track_infos']['trajs']
        object_id = scene_data['track_infos']['object_id']
        object_type = scene_data['track_infos']['object_type']
        gt_is_valid = (gt_trajs[:, :, -1] > 0).astype(int)  # `valid` mask
        tracks_to_predict = scene_data['tracks_to_predict']['track_index']
        # 若为规划赛项使用
        # tracks_to_predict = scene_data['sdc_track_index']

        # 将提取的信息存入字典
        gt_data[scenario_id] = {
            'gt_trajs': gt_trajs, # 所有轨迹
            'object_id': object_id,
            'object_type': object_type,
            'gt_is_valid': gt_is_valid,
            'tracks_to_predict':tracks_to_predict
        }
        
    # 保存 `gt_data.pkl`
    with open(output_file, 'wb') as f:
        pickle.dump(gt_data, f)
    print(f"GT data saved to {output_file}")

# 使用示例
processed_dir = './data/waymo/processed_scenarios_for_competition'
output_file = './PATH/TO/GT/gt_data.pkl'
create_gt_data_from_processed_files(processed_dir, output_file)
