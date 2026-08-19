#!/usr/bin/env bash
# 构建提交镜像：打包 conda 环境 → 生成 result.pkl → 生成演示图片 → 构建 Docker 镜像 → 导出 Tar
# 在远程训练机 /root/wanqinghua/MTR 目录下执行
set -ex

IMAGE_NAME="mtr-voyah-submission"
TAR_NAME="mtr_voyah_submission.tar"
MTR_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$MTR_DIR"

echo "=== 0. 检查前置文件 ==="

# 确保模型权重存在
if [ ! -f model/best_model_ema.pth ]; then
    echo "model/best_model_ema.pth 不存在，从训练输出拷贝..."
    cp output/waymo/mtr_voyah_data/mtr_with_qcnet_v1/ckpt/best_model_ema.pth model/best_model_ema.pth
fi
ls -lh model/best_model_ema.pth

# 确保 conda 环境包存在
if [ ! -f mtr.tar.gz ]; then
    echo "mtr.tar.gz 不存在，执行 pack_env.sh 打包 conda 环境..."
    bash pack_env.sh
fi
ls -lh mtr.tar.gz

# 确保测试集推理结果 result.pkl 存在（在大测试集上推理生成）
RESULT_PKL="output/result.pkl"
if [ ! -f "$RESULT_PKL" ]; then
    echo "$RESULT_PKL 不存在，在大测试集上执行推理..."
    python tools/test.py \
        --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
        --ckpt model/best_model_ema.pth \
        --extra_tag submission \
        --batch_size 64 \
        --workers 8 \
        --save_to_file \
        --set DATA_CONFIG.DATA_ROOT ../data \
              DATA_CONFIG.SPLIT_DIR.test processed_scenarios_testing_B1_part \
              DATA_CONFIG.INFO_FILE.test processed_scenarios_testB1_part_infos.pkl
    # 找到生成的 result.pkl 并拷贝到 output/
    GENERATED=$(find output -name "result.pkl" -path "*submission*" | head -1)
    if [ -z "$GENERATED" ]; then
        GENERATED=$(find output -name "result.pkl" | head -1)
    fi
    cp "$GENERATED" "$RESULT_PKL"
fi
ls -lh "$RESULT_PKL"

echo "=== 0.5. 生成演示图片 ==="
# 在验证集上推理（有真值，用于可视化对比）
EVAL_RESULT_PKL="output/eval_result.pkl"
if [ ! -f "$EVAL_RESULT_PKL" ]; then
    echo "在验证集上推理生成带真值的结果..."
    python tools/test.py \
        --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
        --ckpt model/best_model_ema.pth \
        --extra_tag demo_vis \
        --batch_size 64 \
        --workers 8 \
        --save_to_file \
        --set DATA_CONFIG.DATA_ROOT ../data \
              DATA_CONFIG.SPLIT_DIR.test processed_scenarios_validation \
              DATA_CONFIG.INFO_FILE.test processed_scenarios_val_infos.pkl
    GENERATED_EVAL=$(find output -name "result.pkl" -path "*demo_vis*" | head -1)
    if [ -z "$GENERATED_EVAL" ]; then
        GENERATED_EVAL=$(find output -name "result.pkl" -path "*eval*" | head -1)
    fi
    if [ -n "$GENERATED_EVAL" ]; then
        cp "$GENERATED_EVAL" "$EVAL_RESULT_PKL"
    fi
fi

# 生成演示图片（取前 3 个场景）
if [ -f "$EVAL_RESULT_PKL" ]; then
    echo "生成演示图片..."
    python tools/visualize_prediction.py \
        --result_pkl "$EVAL_RESULT_PKL" \
        --data_dir ../data/processed_scenarios_validation \
        --output_dir 演示图片 \
        --all || true
    # 如果 --all 太多，只保留前 3 张
    mkdir -p 演示图片
    ls -t 演示图片/*.png 2>/dev/null | head -3 > /tmp/keep_images.txt || true
    # 如果没生成图片，尝试单个场景
    if [ ! -s /tmp/keep_images.txt ]; then
        FIRST_SCENARIO=$(python -c "
import pickle
with open('$EVAL_RESULT_PKL','rb') as f: r=pickle.load(f)
s = r[0]['scenario_id'] if isinstance(r, list) else list(r.keys())[0]
print(s)
" 2>/dev/null || echo "")
        if [ -n "$FIRST_SCENARIO" ]; then
            python tools/visualize_prediction.py \
                --result_pkl "$EVAL_RESULT_PKL" \
                --data_dir ../data/processed_scenarios_validation \
                --scenario_id "$FIRST_SCENARIO" \
                --output_dir 演示图片 || true
        fi
    fi
    ls -lh 演示图片/*.png 2>/dev/null || echo "警告: 未生成演示图片"
fi

echo "=== 1. 构建 Docker 镜像 ==="
docker build -t ${IMAGE_NAME} .

echo "=== 2. 导出 Docker 镜像为 Tar 包 ==="
docker save -o ${TAR_NAME} ${IMAGE_NAME}

echo "=== 3. 验证镜像 ==="
docker run --rm ${IMAGE_NAME} python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== 完成 ==="
echo "镜像 Tar 包: ${TAR_NAME}"
echo "镜像名称: ${IMAGE_NAME}"
ls -lh ${TAR_NAME}
