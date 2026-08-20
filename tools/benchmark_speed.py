"""性能测速脚本：测量单 batch 平均推理耗时

用法:
    python benchmark_speed.py --cfg_file cfgs/waymo/mtr_voyah_data.yaml --ckpt /workspace/model/best_model.pth
"""
import argparse
import logging
import time
import torch

from mtr.models.model import MotionTransformer
from mtr.config import cfg, cfg_from_yaml_file


def main():
    parser = argparse.ArgumentParser(description="MTR 推理性能测速")
    parser.add_argument("--cfg_file", type=str, default="cfgs/waymo/mtr_voyah_data.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--num_runs", type=int, default=10, help="推理次数")
    args = parser.parse_args()

    logger = logging.getLogger("benchmark")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg_from_yaml_file(args.cfg_file, cfg)
    model = MotionTransformer(config=cfg.MODEL).cuda().eval()
    model.load_params_with_optimizer(args.ckpt, to_cpu=False, logger=logger)

    # 构造随机输入测速
    batch_dict = {
        "input_dict": {
            "scenario_id": ["speed_test"],
            "obj_trajs": torch.randn(1, 64, 11, 10).cuda(),
            "obj_trajs_mask": torch.ones(1, 64, 11).cuda(),
            "map_polylines": torch.randn(1, 768, 20, 9).cuda(),
            "map_polylines_mask": torch.ones(1, 768, 20).cuda(),
            "track_index_to_predict": torch.tensor([0]).cuda(),
            "obj_types": [["TYPE_VEHICLE"]],
            "object_id": torch.tensor([[0]]).cuda(),
        }
    }

    # warmup
    with torch.no_grad():
        model(batch_dict)
    torch.cuda.synchronize()

    # 正式测速
    t0 = time.time()
    for _ in range(args.num_runs):
        with torch.no_grad():
            model(batch_dict)
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) / args.num_runs * 1000
    print(f"平均推理耗时: {avg_ms:.1f} ms/batch ({args.num_runs} 次平均)")


if __name__ == "__main__":
    main()
