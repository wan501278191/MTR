"""性能测速脚本：使用真实数据测量单 batch 平均推理耗时

用法:
    python benchmark_speed.py --cfg_file cfgs/waymo/mtr_voyah_data.yaml --ckpt /workspace/model/best_model.pth --data_root /mnt/data
"""
import argparse
import logging
import time
from pathlib import Path

import torch

from mtr.models.model import MotionTransformer
from mtr.config import cfg, cfg_from_yaml_file
from mtr.datasets import build_dataloader


def main():
    parser = argparse.ArgumentParser(description="MTR 推理性能测速")
    parser.add_argument("--cfg_file", type=str, default="cfgs/waymo/mtr_voyah_data.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--num_runs", type=int, default=10, help="推理次数")
    parser.add_argument("--batch_size", type=int, default=1, help="测速 batch 大小")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--data_root", type=str, default=None, help="覆盖数据集根目录")
    args = parser.parse_args()

    logger = logging.getLogger("benchmark")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.data_root:
        cfg.DATA_CONFIG.DATA_ROOT = args.data_root
        logger.info(f"DATA_ROOT 覆盖为: {args.data_root}")

    # 用验证集加载一个真实 batch
    test_set, test_loader, _ = build_dataloader(
        cfg.DATA_CONFIG, args.batch_size, dist=False, workers=args.workers, logger=logger, training=False
    )

    model = MotionTransformer(config=cfg.MODEL).cuda().eval()
    model.load_params_with_optimizer(args.ckpt, to_cpu=False, logger=logger)

    # 取一个真实 batch（模型内部会将 tensor 移至 GPU）
    batch_dict = next(iter(test_loader))

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
    print(f"平均推理耗时: {avg_ms:.1f} ms/batch ({args.num_runs} 次平均, batch_size={args.batch_size})")


if __name__ == "__main__":
    main()
