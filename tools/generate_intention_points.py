"""
Generate intention points from GT trajectory endpoints with percentile clipping.

Usage (on server with data):
    cd /root/wanqinghua/MTR/tools
    python generate_intention_points.py \
        --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
        --percentile 90 \
        --num_clusters 64 \
        --output ../data/waymo/cluster_64_center_dict.pkl
"""
import argparse
import pickle
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mtr.config import cfg, cfg_from_yaml_file
from mtr.datasets import build_dataloader


def collect_gt_endpoints(cfg):
    """Collect GT final endpoints from training set."""
    logger = logging.getLogger('gen_intention')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=True,
    )

    class_names = cfg.DATA_CONFIG.OBJECT_TYPE
    endpoints = {name: [] for name in class_names}
    total = len(train_set)

    for idx in range(total):
        data = train_set[idx]
        obj_types = data['center_objects_type']
        gt_trajs = data['center_gt_trajs']          # (num_center_objects, num_future_timestamps, 4)
        gt_mask = data['center_gt_trajs_mask']      # (num_center_objects, num_future_timestamps)
        center_gt_final_valid_idx = data['center_gt_final_valid_idx']

        for i, obj_type in enumerate(obj_types):
            final_idx = int(center_gt_final_valid_idx[i])
            if final_idx < 0 or gt_mask[i, final_idx] == 0:
                continue
            endpoint = gt_trajs[i, final_idx, 0:2]
            endpoints[str(obj_type)].append(endpoint)

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx+1} / {total} scenarios")

    print(f"  Total {total} scenarios, endpoints collected:")
    for k, v in endpoints.items():
        print(f"    {k}: {len(v)} endpoints")
    return {k: np.array(v, dtype=np.float32) for k, v in endpoints.items()}


def cluster_with_clip(endpoints, percentile=90, num_clusters=64):
    """K-means on percentile-clipped endpoints."""
    result = {}
    for obj_type, pts in endpoints.items():
        if len(pts) < num_clusters:
            print(f"  WARN: {obj_type} has only {len(pts)} endpoints, < {num_clusters} clusters, using all")
            result[obj_type] = pts
            continue

        lower = np.percentile(pts, 100 - percentile, axis=0)
        upper = np.percentile(pts, percentile, axis=0)
        clipped = np.clip(pts, lower, upper)

        km = KMeans(n_clusters=num_clusters, random_state=666, n_init=10)
        km.fit(clipped)
        result[obj_type] = km.cluster_centers_.astype(np.float32)

        print(f"  {obj_type}:")
        print(f"    raw_n={len(pts)}, clip=[{100-percentile:.0f}%, {percentile:.0f}%]")
        print(f"    clip range: x[{lower[0]:.1f}, {upper[0]:.1f}] y[{lower[1]:.1f}, {upper[1]:.1f}]")
        print(f"    centers:    x[{result[obj_type][:,0].min():.1f}, {result[obj_type][:,0].max():.1f}] y[{result[obj_type][:,1].min():.1f}, {result[obj_type][:,1].max():.1f}]")
        print(f"    std:        x={result[obj_type][:,0].std():.1f} y={result[obj_type][:,1].std():.1f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo/mtr_voyah_data.yaml')
    parser.add_argument('--percentile', type=float, default=90.0,
                        help='Clip to [100-p, p] percentile (90=remove top/bottom 10%% outliers)')
    parser.add_argument('--num_clusters', type=int, default=64)
    parser.add_argument('--output', type=str, default='../data/waymo/cluster_64_center_dict.pkl')
    args = parser.parse_args()

    # Resolve cfg_file relative to tools/ directory (where this script lives)
    script_dir = Path(__file__).resolve().parent
    cfg_path = script_dir / args.cfg_file
    if not cfg_path.exists():
        # try relative to project root
        cfg_path = script_dir.parent / args.cfg_file
    if not cfg_path.exists():
        cfg_path = Path(args.cfg_file)

    cfg_from_yaml_file(cfg_path, cfg)

    print(f"Step 1: Collecting GT endpoints (percentile={args.percentile})...")
    endpoints = collect_gt_endpoints(cfg)

    print(f"\nStep 2: K-means clustering to {args.num_clusters} centers with percentile clip...")
    result = cluster_with_clip(endpoints, percentile=args.percentile, num_clusters=args.num_clusters)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = script_dir / args.output
    if out_path.exists():
        bak = out_path.with_suffix('.pkl.bak')
        if not bak.exists():
            import shutil
            shutil.copy(out_path, bak)
            print(f"\n  Backed up existing file to {bak}")

    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nDone! Saved to {out_path}")


if __name__ == '__main__':
    main()
