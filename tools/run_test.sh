#!/usr/bin/env bash
# 运行测试 + 生成预测结果 + 可视化动图
# 用法: bash run_test.sh [scenario_id] [object_index] [future_seconds]

set -e
cd "$(dirname "$0")"

# ==================== 配置区（按需修改） ====================
CFG_FILE="cfgs/waymo/mtr_voyah_data.yaml"   # 配置文件
EXTRA_TAG="baseline"                         # 实验标签（输出目录名）
TEST_TAG="test_with_train"                   # 测试标签（test/ 下的子目录名）
OUTPUT_DIR=""                                 # 可视化输出目录（留空自动绑定到 result.pkl 同级）
DATA_DIR="../../data/processed_scenarios_testing_A_full"  # 场景数据目录（可视化用）
# 测试集数据路径（相对于 DATA_ROOT 的子目录名和 infos 文件名）
# 留空则使用 yaml 中 SPLIT_DIR.test / INFO_FILE.test 的默认值（验证集）
TEST_SPLIT_DIR="processed_scenarios_testing_A_full"
TEST_INFO_FILE="processed_scenarios_testA_full_infos.pkl"
# ===========================================================

# 命令行参数（可覆盖默认值）
SCENARIO_ID="${1:-enc_00ab2662a208ecc4a140e9a9}"
OBJECT_INDEX="${2:-0}"
FUTURE_SECONDS="${3:-8.0}"

# ---- 从 CFG_FILE / EXTRA_TAG / TEST_TAG 推导所有路径 ----
CFG_TAG=$(basename "$CFG_FILE" .yaml)                  # e.g. mtr_voyah_data
EXP_GROUP_PATH=$(dirname "$CFG_FILE" | xargs basename)  # e.g. waymo
BASE_DIR="../output/${EXP_GROUP_PATH}/${CFG_TAG}/${EXTRA_TAG}"
CKPT_DIR="${BASE_DIR}/ckpt"
TEST_BASE="${BASE_DIR}/test"

# 选择 checkpoint：优先 best_model，回退到最新 checkpoint
BEST_CKPT="${CKPT_DIR}/best_model.pth"
BEST_RECORD="${TEST_BASE}/${TEST_TAG}/best_eval_record.txt"

if [ -f "$BEST_CKPT" ]; then
    CKPT="$BEST_CKPT"
    EPOCH_NUM=$(grep -oP 'best_epoch_\K\d+' "$BEST_RECORD" 2>/dev/null | tail -1)
    if [ -z "$EPOCH_NUM" ]; then
        EPOCH_NUM=$(ls -t ${CKPT_DIR}/checkpoint_epoch_*.pth 2>/dev/null | head -1 | grep -oP 'epoch_\K\d+')
    fi
    echo "Using best_model (epoch ${EPOCH_NUM})"
elif [ -n "$(ls -t ${CKPT_DIR}/checkpoint_epoch_*.pth 2>/dev/null | head -1)" ]; then
    CKPT=$(ls -t ${CKPT_DIR}/checkpoint_epoch_*.pth 2>/dev/null | head -1)
    EPOCH_NUM=$(echo "$CKPT" | grep -oP 'epoch_\K\d+')
    echo "Using latest checkpoint (epoch ${EPOCH_NUM})"
else
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

# result.pkl 来源：
#   独立 test.py 推理（数字ckpt）: test/epoch_{N}/{TEST_TAG}/result.pkl
#   独立 test.py 推理（best_model）: test/{TEST_TAG}/result.pkl
# 训练时的验证集评估结果在 eval/ 目录下，与此处的 test/ 目录隔离，不会冲突。

# 查找已有的 test.py 推理 result.pkl（两种结构都搜，取最新）
RESULT_PKL=$(find "${TEST_BASE}" -path "*/${TEST_TAG}/result.pkl" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | awk '{print $2}')

# 可视化输出目录：用户指定优先，否则绑定到 result.pkl 同级目录
if [ -z "$OUTPUT_DIR" ] && [ -n "$RESULT_PKL" ]; then
    OUTPUT_DIR="$(dirname "$RESULT_PKL")/visualization"
fi
[ -n "$OUTPUT_DIR" ] && mkdir -p "$OUTPUT_DIR"

echo "========== 配置 =========="
echo "  CFG_FILE:   $CFG_FILE"
echo "  EXTRA_TAG:  $EXTRA_TAG"
echo "  TEST_TAG:   $TEST_TAG"
echo "  CKPT:       $CKPT"
echo "  RESULT:     ${RESULT_PKL:-(将由 Step 1 生成)}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR:-(将由 Step 1 后自动绑定)}"
echo "  DATA_DIR:   $DATA_DIR"
echo "  SCENARIO:   $SCENARIO_ID (obj=$OBJECT_INDEX, ${FUTURE_SECONDS}s)"
echo "=========================="

# Step 1: 如果没有测试集 result.pkl，运行推理生成
if [ -z "$RESULT_PKL" ]; then
    echo ""
    echo "========== 1. 运行测试生成预测结果 =========="
    python test.py \
        --cfg_file "$CFG_FILE" \
        --extra_tag "$EXTRA_TAG" \
        --ckpt "$CKPT" \
        --test_tag "$TEST_TAG" \
        --scenario_id "$SCENARIO_ID" \
        ${TEST_SPLIT_DIR:+--test_split_dir "$TEST_SPLIT_DIR"} \
        ${TEST_INFO_FILE:+--test_info_file "$TEST_INFO_FILE"} \
        --max_waiting_mins 0

    # 推理后动态查找 result.pkl（两种结构都搜，取最新）
    RESULT_PKL=$(find "${TEST_BASE}" -path "*/${TEST_TAG}/result.pkl" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')

    if [ -z "$RESULT_PKL" ]; then
        echo "ERROR: result.pkl not found after inference"
        find "${TEST_BASE}" -name "result.pkl" 2>/dev/null
        exit 1
    fi

    # 推理后才确定 result.pkl 位置，绑定 OUTPUT_DIR
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$(dirname "$RESULT_PKL")/visualization"
        mkdir -p "$OUTPUT_DIR"
    fi

    echo "Inference result: $RESULT_PKL"
else
    echo ""
    echo "========== 1. 跳过推理（已有 result.pkl） =========="
fi

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
