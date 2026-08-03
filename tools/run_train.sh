#!/usr/bin/env bash
# 本地smoke训练 — 删除旧checkpoint后从头训练
set -e
cd "$(dirname "$0")"

EXTRA_TAG="${1:-local_smoke}"
EPOCHS="${2:-30}"
BATCH_SIZE="${3:-1}"

CKPT_DIR="../output/waymo/mtr_voyah_smoke/${EXTRA_TAG}/ckpt"

echo "========== 清理旧checkpoint =========="
rm -rf "${CKPT_DIR}"
echo "已清理: ${CKPT_DIR}"

echo ""
echo "========== 开始训练 (epochs=${EPOCHS}, batch_size=${BATCH_SIZE}) =========="
python train.py \
    --cfg_file cfgs/waymo/mtr_voyah_smoke.yaml \
    --extra_tag "${EXTRA_TAG}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --workers 0 \
    --not_eval_with_train
