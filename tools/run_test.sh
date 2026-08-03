#!/usr/bin/env bash
# 运行测试 + 生成预测结果 + 可视化动图
# 用法: bash run_test.sh [scenario_id] [object_index] [future_seconds]

set -e
cd "$(dirname "$0")"

SCENARIO_ID="${1:-enc_00ab2662a208ecc4a140e9a9}"
OBJECT_INDEX="${2:-0}"
FUTURE_SECONDS="${3:-8.0}"
CKPT="../output/waymo/mtr_voyah_smoke/local_smoke/ckpt/checkpoint_epoch_30.pth"
RESULT_PKL="../output/waymo/mtr_voyah_smoke/local_smoke/eval/epoch_30/smoke_testA/result.pkl"
DATA_DIR="../../data/processed_scenarios_testing_A_full"
OUTPUT_DIR="../output/visualization"

# 自动查找最新checkpoint
LATEST_CKPT=$(ls -t ../output/waymo/mtr_voyah_smoke/local_smoke/ckpt/checkpoint_epoch_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    CKPT="$LATEST_CKPT"
    EPOCH_NUM=$(echo "$CKPT" | grep -oP 'epoch_\K\d+')
    RESULT_PKL="../output/waymo/mtr_voyah_smoke/local_smoke/eval/epoch_${EPOCH_NUM}/smoke_testA/result.pkl"
fi

echo "========== 1. 运行测试生成预测结果 =========="
echo "Checkpoint: $CKPT"
python test.py \
    --cfg_file cfgs/waymo/mtr_voyah_smoke.yaml \
    --extra_tag local_smoke \
    --ckpt "$CKPT" \
    --eval_tag smoke_testA \
    --max_waiting_mins 0

echo ""
echo "========== 2. 生成静态可视化 PNG =========="
python visualize_prediction.py \
    --result_pkl "$RESULT_PKL" \
    --data_dir "$DATA_DIR" \
    --scenario_id "$SCENARIO_ID" \
    --object_index "$OBJECT_INDEX" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========== 3. 生成未来 ${FUTURE_SECONDS}s 动图 GIF =========="
python visualize_animation.py \
    --result_pkl "$RESULT_PKL" \
    --data_dir "$DATA_DIR" \
    --scenario_id "$SCENARIO_ID" \
    --object_index "$OBJECT_INDEX" \
    --output_dir "$OUTPUT_DIR" \
    --fps 10 \
    --margin 50 \
    --future_seconds "$FUTURE_SECONDS"

echo ""
echo "========== 完成 =========="
echo "静态图: $OUTPUT_DIR/visualization_${SCENARIO_ID}_*.png"
echo "动图:   $OUTPUT_DIR/animation_${SCENARIO_ID}_*_${FUTURE_SECONDS}s.gif"
