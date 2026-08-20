#!/usr/bin/env bash
# 构建提交镜像：打包 conda 环境 → 生成 result.pkl → 生成演示图片 → 构建 Docker 镜像 → 导出 Tar
# 在远程训练机 /root/wanqinghua/MTR 目录下执行
set -ex

IMAGE_NAME="mtr-voyah-submission"
TAR_NAME="mtr_voyah_submission.tar.gz"
MTR_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$MTR_DIR"

echo "=== 0. 检查前置文件 ==="

# 确保模型权重存在
if [ ! -f model/best_model.pth ]; then
    echo "错误: model/best_model.pth 不存在，请将训练好的模型权重放到 model/best_model.pth"
    exit 1
fi
ls -lh model/best_model.pth

# 确保 conda 环境包存在
if [ ! -f mtr.tar.gz ]; then
    echo "mtr.tar.gz 不存在，执行 pack_env.sh 打包 conda 环境..."
    bash pack_env.sh
fi
ls -lh mtr.tar.gz

# === 0.5. 清空输出目录 ===
echo "=== 0.5. 清空输出目录 ==="
rm -rf test_output 演示图片
mkdir -p test_output 演示图片/eval 演示图片/test

# === 1. 测试集全量推理 → test_output/result.pkl ===
echo "=== 1. 测试集全量推理 ==="
RESULT_PKL="test_output/result.pkl"
cd tools
python test.py \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --ckpt ../model/best_model.pth \
    --extra_tag submission \
    --batch_size 32 \
    --workers 8 \
    --save_to_file \
    --set DATA_CONFIG.SPLIT_DIR.test processed_scenarios_testing_B1_part \
          DATA_CONFIG.INFO_FILE.test processed_scenarios_testB1_part_infos.pkl
cd "$MTR_DIR"
# test.py 输出到 output/ 目录，从中找到 result.pkl 拷贝到 test_output/
GENERATED=$(find output -name "result.pkl" -path "*submission*" | head -1)
if [ -z "$GENERATED" ]; then
    GENERATED=$(find output -name "result.pkl" | head -1)
fi
if [ -z "$GENERATED" ]; then
    echo "错误: 推理后未找到 result.pkl"
    exit 1
fi
cp "$GENERATED" "$RESULT_PKL"
ls -lh "$RESULT_PKL"

# === 2. 验证集推理（有真值，用于演示对比）→ test_output/eval_result.pkl ===
echo "=== 2. 验证集推理 ==="
EVAL_RESULT_PKL="test_output/eval_result.pkl"
cd tools
python test.py \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --ckpt ../model/best_model.pth \
    --extra_tag demo_vis \
    --batch_size 32 \
    --workers 8 \
    --save_to_file \
    --set DATA_CONFIG.SPLIT_DIR.test processed_scenarios_validation \
          DATA_CONFIG.INFO_FILE.test processed_scenarios_val_infos.pkl
cd "$MTR_DIR"
GENERATED_EVAL=$(find output -name "result.pkl" -path "*demo_vis*" | head -1)
if [ -z "$GENERATED_EVAL" ]; then
    GENERATED_EVAL=$(find output -name "result.pkl" -path "*eval*" | head -1)
fi
if [ -n "$GENERATED_EVAL" ]; then
    cp "$GENERATED_EVAL" "$EVAL_RESULT_PKL"
fi
ls -lh "$EVAL_RESULT_PKL" 2>/dev/null || echo "警告: eval_result.pkl 不存在"

# === 3. 生成演示图片（eval + test 各 3 张）===
echo "=== 3. 生成演示图片 ==="
cd tools

# eval（验证集，有真值对比）
if [ -f "../$EVAL_RESULT_PKL" ]; then
    echo "生成 eval 演示图片..."
    python visualize_prediction.py \
        --result_pkl "../$EVAL_RESULT_PKL" \
        --data_dir ../../data/processed_scenarios_validation \
        --output_dir ../演示图片/eval \
        --all --max_scenarios 3 || true
fi

# test（大测试集）
if [ -f "../$RESULT_PKL" ]; then
    echo "生成 test 演示图片..."
    python visualize_prediction.py \
        --result_pkl "../$RESULT_PKL" \
        --data_dir ../../data/processed_scenarios_testing_B1_part \
        --output_dir ../演示图片/test \
        --all --max_scenarios 3 || true
fi

cd "$MTR_DIR"
echo "=== 演示图片列表 ==="
find 演示图片 -name "*.png" -exec ls -lh {} \; 2>/dev/null || echo "警告: 未生成演示图片"

# === 4. 清理 Docker 并构建镜像 ===
echo "=== 4. 清理 Docker 无用资源 ==="
docker system prune -af

echo "=== 4.5. 加载本地 CUDA 基础镜像 ==="
CUDA_TAR=$(find /root -maxdepth 3 -name "cuda11.8*amd64.tar" 2>/dev/null | head -1)
if [ -n "$CUDA_TAR" ]; then
    echo "加载 $CUDA_TAR ..."
    docker load -i "$CUDA_TAR"
else
    echo "未找到本地 cuda11.8 tar 包，尝试直接构建（需要网络）..."
fi

echo "=== 5. 构建 Docker 镜像（多阶段构建）==="
docker build -t ${IMAGE_NAME} .

echo "=== 5.5 镜像大小 ==="
docker images ${IMAGE_NAME} --format "镜像: {{.Repository}}:{{.Tag}}  大小: {{.Size}}"

echo "=== 6. 导出 Docker 镜像为 Tar 包（gzip 压缩）==="
docker save ${IMAGE_NAME} | gzip > ${TAR_NAME}
ls -lh ${TAR_NAME}
echo "压缩前镜像大小:"
docker images ${IMAGE_NAME} --format "{{.Size}}"
echo "压缩后 tar 包大小:"
ls -lh ${TAR_NAME} | awk '{print $5}'

echo "=== 7. 镜像构建后自动验证操作手册 ==="
echo ""
echo "--- 7.1 验证镜像环境 (import torch / import mtr) ---"
docker run --rm --shm-size=8g ${IMAGE_NAME} python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
docker run --rm --shm-size=8g ${IMAGE_NAME} python -c "import mtr; print('import mtr OK')"

echo ""
echo "--- 7.2 需求1: 模型推理 (验证 result.pkl 输出到 /mnt/output) ---"
rm -rf /tmp/mtr_verify_output
mkdir -p /tmp/mtr_verify_output
docker run --gpus all --rm --shm-size=8g \
    -v $(pwd)/../data:/mnt/data \
    -v /tmp/mtr_verify_output:/mnt/output \
    ${IMAGE_NAME}
if [ -f /tmp/mtr_verify_output/result.pkl ]; then
    echo "✅ 需求1 通过: /mnt/output/result.pkl 已生成"
    ls -lh /tmp/mtr_verify_output/result.pkl
else
    echo "❌ 需求1 失败: /mnt/output/result.pkl 不存在"
    exit 1
fi

echo ""
echo "--- 7.3 需求3: 结果验证 (verify_result.py) ---"
docker run --gpus all --rm --shm-size=8g \
    -v /tmp/mtr_verify_output:/mnt/output \
    --entrypoint /bin/bash \
    ${IMAGE_NAME} \
    -c "cd /workspace/MTR/tools && python verify_result.py --result_pkl /mnt/output/result.pkl"

echo ""
echo "--- 7.4 需求4: 性能测速 (单 batch 推理耗时) ---"
docker run --gpus all --rm --shm-size=8g \
    -v /tmp/mtr_verify_output:/mnt/output \
    --entrypoint /bin/bash \
    ${IMAGE_NAME} \
    -c 'cd /workspace/MTR/tools && python -c "
import time, torch
from mtr.models.model import MotionTransformer
from mtr.config import cfg, cfg_from_yaml_file
cfg_from_yaml_file("cfgs/waymo/mtr_voyah_data.yaml", cfg)
model = MotionTransformer(cfg).cuda().eval()
model.load_params_with_optimizer("/workspace/model/best_model.pth", to_cpu=False)
batch = {"input_dict": {"scenario_id": ["speed_test"], "obj_trajs": torch.randn(1, 64, 11, 10).cuda()}}
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    with torch.no_grad():
        model(batch)
torch.cuda.synchronize()
t1 = time.time()
print(f"平均推理耗时: {(t1-t0)/5*1000:.1f} ms/batch")
"'

echo ""
echo "--- 7.5 需求5: 性能评估 (get_gt_data.py + eval_gt_and_pred.py) ---"
# 生成 GT 数据
docker run --gpus all --rm --shm-size=8g \
    -v $(pwd)/../data:/mnt/data \
    -v /tmp/mtr_verify_output:/mnt/output \
    --entrypoint /bin/bash \
    ${IMAGE_NAME} \
    -c "cd /workspace/MTR/tools/eval_scripts && python get_gt_data.py \
        --processed_dir /mnt/data/processed_scenarios_validation \
        --output_file /mnt/output/gt_data.pkl"
# 评估
docker run --gpus all --rm --shm-size=8g \
    -v /tmp/mtr_verify_output:/mnt/output \
    --entrypoint /bin/bash \
    ${IMAGE_NAME} \
    -c "cd /workspace/MTR/tools/eval_scripts && python eval_gt_and_pred.py \
        --pred_file /mnt/output/result.pkl \
        --gt_file /mnt/output/gt_data.pkl \
        --eval_second 3"

echo ""
echo "--- 7.6 需求6: 单条结果推理 (小测试集 smoke test) ---"
docker run --gpus all --rm --shm-size=8g \
    -v $(pwd)/../data:/mnt/data \
    -v /tmp/mtr_verify_smoke:/mnt/output \
    --entrypoint /bin/bash \
    ${IMAGE_NAME} \
    -c "cd /workspace/MTR/tools && python test.py \
        --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
        --ckpt /workspace/model/best_model.pth \
        --extra_tag smoke_test \
        --batch_size 4 \
        --workers 4 \
        --save_to_file \
        --set DATA_CONFIG.SPLIT_DIR.test processed_scenarios_validation \
              DATA_CONFIG.INFO_FILE.test processed_scenarios_val_infos.pkl"

echo ""
echo "=== 8. 验证完成 ==="
echo "所有操作手册命令验证通过:"
echo "  ✅ 7.1 镜像环境 (torch + mtr)"
echo "  ✅ 7.2 需求1 模型推理 (result.pkl → /mnt/output)"
echo "  ✅ 7.3 需求3 结果验证 (verify_result.py)"
echo "  ✅ 7.4 需求4 性能测速"
echo "  ✅ 7.5 需求5 性能评估 (get_gt_data + eval_gt_and_pred)"
echo "  ✅ 7.6 需求6 单条推理"

echo ""
echo "=== 完成 ==="
echo "镜像 Tar 包: ${TAR_NAME}"
echo "镜像名称: ${IMAGE_NAME}"
ls -lh ${TAR_NAME}

# 清理验证临时目录
rm -rf /tmp/mtr_verify_output /tmp/mtr_verify_smoke
