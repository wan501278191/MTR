"""验证 result.pkl 格式是否符合赛事要求。

用法:
    python verify_result.py --result_pkl path/to/result.pkl

验证项:
    - 结构: list[list[dict]]
    - 每个 dict 含: scenario_id, pred_trajs(6,80,2), pred_scores(6,), object_id, object_type, track_index_to_predict
    - pred_scores sum ≈ 1.0
    - 无 NaN
"""
import argparse
import pickle
import sys

import numpy as np


def verify_result(result_path, verbose=True):
    with open(result_path, "rb") as f:
        data = pickle.load(f)

    errors = []
    warnings = []

    if not isinstance(data, list):
        errors.append(f"顶层类型错误: 期望 list, 得到 {type(data)}")
        return errors, warnings

    total_objects = 0
    for i, scene in enumerate(data):
        if not isinstance(scene, list):
            errors.append(f"场景 {i}: 期望 list, 得到 {type(scene)}")
            continue

        for j, obj in enumerate(scene):
            total_objects += 1
            required_keys = ["scenario_id", "pred_trajs", "pred_scores", "object_id", "object_type", "track_index_to_predict"]
            for key in required_keys:
                if key not in obj:
                    errors.append(f"场景{i} 物体{j}: 缺少 key '{key}'")

            trajs = obj.get("pred_trajs")
            if trajs is not None:
                if trajs.shape != (6, 80, 2):
                    errors.append(f"场景{i} 物体{j}: pred_trajs shape={trajs.shape}, 期望(6,80,2)")
                if np.isnan(trajs).any():
                    errors.append(f"场景{i} 物体{j}: pred_trajs 含 NaN")

            scores = obj.get("pred_scores")
            if scores is not None:
                if scores.shape != (6,):
                    errors.append(f"场景{i} 物体{j}: pred_scores shape={scores.shape}, 期望(6,)")
                s = scores.sum()
                if abs(s - 1.0) > 0.01:
                    warnings.append(f"场景{i} 物体{j}: pred_scores sum={s:.4f} (期望≈1.0)")

    if verbose:
        print(f"场景数: {len(data)}")
        print(f"预测目标总数: {total_objects}")
        if errors:
            print(f"\n❌ 错误 ({len(errors)}):")
            for e in errors[:10]:
                print(f"  - {e}")
            if len(errors) > 10:
                print(f"  ... 还有 {len(errors)-10} 个")
        else:
            print("\n✅ 格式验证通过")
        if warnings:
            print(f"\n⚠️  警告 ({len(warnings)}):")
            for w in warnings[:5]:
                print(f"  - {w}")

    return errors, warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_pkl", required=True, help="result.pkl 路径")
    args = parser.parse_args()
    errors, _ = verify_result(args.result_pkl)
    sys.exit(1 if errors else 0)
