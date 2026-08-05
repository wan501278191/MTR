#!/usr/bin/env bash
# 运行测试 + 生成预测结果 + 可视化动图
# 用法: bash run_test.sh [scenario_id] [object_index] [future_seconds]

set -e
cd "$(dirname "$0")"

# ==================== 配置区（按需修改） ====================
CFG_FILE="cfgs/waymo/mtr_voyah_smoke.yaml"       # 配置文件（smoke=冒烟测试, data=正式训练）
EXTRA_TAG="local_smoke"                            # 实验标签（输出目录名）
EVAL_TAG="smoke_testA"                             # 评估标签
OUTPUT_DIR="../output/visualization"               # 可视化输出目录
DATA_DIR="../../data/processed_scenarios_testing_A_full"  # 场景数据目录
# ===========================================================

# 命令行参数（可覆盖默认值）
SCENARIO_ID="${1:-enc_00ab2662a208ecc4a140e9a9}"
OBJECT_INDEX="${2:-0}"
FUTURE_SECONDS="${3:-8.0}"

# 从配置推导 checkpoint 和 result.pkl 路径
CKPT_DIR="../output/waymo/$(basename $(dirname $CFG_FILE))/${EXTRA_TAG}/ckpt"
LATEST_CKPT=$(ls -t ${CKPT_DIR}/checkpoint_epoch_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    CKPT="$LATEST_CKPT"
    EPOCH_NUM=$(echo "$CKPT" | grep -oP 'epoch_\K\d+')
    RESULT_PKL="../output/waymo/$(basename $(dirname $CFG_FILE))/${EXTRA_TAG}/eval/epoch_${EPOCH_NUM}/${EVAL_TAG}/result.pkl"
else
    # 回退到硬编码冒烟测试路径
    CKPT="../output/waymo/mtr_voyah_smoke/local_smoke/ckpt/checkpoint_epoch_30.pth"
    RESULT_PKL="../output/waymo/mtr_voyah_smoke/local_smoke/eval/epoch_30/smoke_testA/result.pkl"
fi

echo "========== 配置 =========="
echo "  CFG_FILE:  $CFG_FILE"
echo "  EXTRA_TAG: $EXTRA_TAG"
echo "  EVAL_TAG:  $EVAL_TAG"
echo "  CKPT:      $CKPT"
echo "  RESULT:    $RESULT_PKL"
echo "  DATA_DIR:  $DATA_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  SCENARIO:  $SCENARIO_ID (obj=$OBJECT_INDEX, ${FUTURE_SECONDS}s)"
echo "=========================="

echo ""
echo "========== 1. 运行测试生成预测结果 =========="
python test.py \
    --cfg_file "$CFG_FILE" \
    --extra_tag "$EXTRA_TAG" \
    --ckpt "$CKPT" \
    --eval_tag "$EVAL_TAG" \
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
