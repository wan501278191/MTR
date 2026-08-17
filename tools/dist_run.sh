#!/usr/bin/env bash
# ==========================================================================
# dist_run.sh — 分布式训练，全部优化已在 mtr_voyah_data.yaml 中开启
#
# 用法:
#   bash dist_run.sh [NGPUS] [BATCH_SIZE] [EPOCHS] [EXTRA_TAG]
#
# 默认: 8 卡 / batch 80 / 30 epoch / extra_tag=optimized
#
# 优化清单 (mtr_voyah_data.yaml):
#   [1] Temperature scaling       — 可学习温度参数 (训练后 calibrate_temperature.py 校准)
#   [2] aWTA Loss                 — 退火 WTA (10→0.1, 30 epoch)
#   [3] Kinematic filter          — 运动学后处理 (速度/加速度/转向约束)
#   [4] TTA                       — 测试时 4 种增强 + 聚类融合 (eval 生效)
#   [5] Dynamic query             — 场景复杂度自适应 intention query (16/64/128)
#   [6] Shared scene encoder      — QCNet 风格因子化注意力 + 极坐标/傅里叶
# ==========================================================================
set -e
cd "$(dirname "$0")"

NGPUS="${1:-8}"
BATCH_SIZE="${2:-80}"
EPOCHS="${3:-30}"
EXTRA_TAG="${4:-optimized}"

CFG_FILE="cfgs/waymo/mtr_voyah_data.yaml"

echo "============================================================"
echo "  MTR++ Optimized Distributed Training"
echo "  GPUs:        ${NGPUS}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Epochs:      ${EPOCHS}"
echo "  Extra tag:   ${EXTRA_TAG}"
echo "  Config:      ${CFG_FILE}"
echo "============================================================"
echo "  Optimizations (in YAML):"
echo "    [1] Temperature scaling       ON (learnable)"
echo "    [2] aWTA loss                 ON (annealed ${EPOCHS} epochs)"
echo "    [3] Kinematic filter          ON (penalize mode)"
echo "    [4] TTA                       ON (eval-time)"
echo "    [5] Dynamic query             ON (16/64/128)"
echo "    [6] Shared scene encoder      ON (factorized + Fourier)"
echo "============================================================"
echo ""

bash scripts/dist_train.sh "${NGPUS}" \
    --cfg_file "${CFG_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --extra_tag "${EXTRA_TAG}"
