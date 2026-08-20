import os
import sys
import pickle
import argparse

def create_gt_data_from_processed_files(processed_dir, output_file, prefix='enc_'):
    """
    从处理后的 Waymo 场景文件中提取 ground truth 轨迹信息，并保存至 gt_data.pkl
    Args:
        processed_dir (str): 已处理文件的目录，包含 `enc_*.pkl` 文件。
        output_file (str): 保存 `gt_data.pkl` 的路径。
    """
    gt_data = {}
    scene_files = [f for f in os.listdir(processed_dir) if f.startswith(prefix) and f.endswith('.pkl')]
    if not scene_files:
        print(f"错误: 在 {processed_dir} 中未找到 {prefix}*.pkl 文件")
        sys.exit(1)

    print(f"找到 {len(scene_files)} 个场景文件")
    for i, scene_file in enumerate(scene_files):
        file_path = os.path.join(processed_dir, scene_file)
        with open(file_path, 'rb') as f:
            scene_data = pickle.load(f)

        scenario_id = scene_data['scenario_id']
        gt_trajs = scene_data['track_infos']['trajs']
        object_id = scene_data['track_infos']['object_id']
        object_type = scene_data['track_infos']['object_type']
        gt_is_valid = (gt_trajs[:, :, -1] > 0).astype(int)
        tracks_to_predict = scene_data['tracks_to_predict']['track_index']

        gt_data[scenario_id] = {
            'gt_trajs': gt_trajs,
            'object_id': object_id,
            'object_type': object_type,
            'gt_is_valid': gt_is_valid,
            'tracks_to_predict': tracks_to_predict
        }
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(scene_files)}")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(gt_data, f)
    print(f"GT data saved to {output_file} ({len(gt_data)} scenarios)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从处理后的场景文件提取 GT 数据')
    parser.add_argument('--processed_dir', type=str, required=True,
                        help='已处理场景文件目录（含 enc_*.pkl）')
    parser.add_argument('--output_file', type=str, default='gt_data.pkl',
                        help='输出 gt_data.pkl 路径')
    args = parser.parse_args()
    create_gt_data_from_processed_files(args.processed_dir, args.output_file)
