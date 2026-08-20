#!/usr/bin/env bash
# 容器入口脚本：执行推理后，将 result.pkl 拷贝到 /mnt/output
set -e

cd /workspace/MTR/tools

# 执行推理
python test.py \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --ckpt /workspace/model/best_model.pth \
    --extra_tag submission \
    --batch_size 32 \
    --workers 8 \
    --save_to_file \
    --set DATA_CONFIG.DATA_ROOT /mnt/data \
          DATA_CONFIG.SPLIT_DIR.test processed_scenarios_testing_B1_part \
          DATA_CONFIG.INFO_FILE.test processed_scenarios_testB1_part_infos.pkl

# 推理结果保存到 output/ 目录，拷贝到 /mnt/output 供评委获取
mkdir -p /mnt/output
RESULT_PKL=$(find /workspace/MTR/output -name "result.pkl" -path "*submission*" | head -1)
if [ -z "$RESULT_PKL" ]; then
    RESULT_PKL=$(find /workspace/MTR/output -name "result.pkl" | head -1)
fi
if [ -z "$RESULT_PKL" ]; then
    echo "错误: 推理后未找到 result.pkl" >&2
    exit 1
fi
cp "$RESULT_PKL" /mnt/output/result.pkl
echo "推理完成，result.pkl 已保存到 /mnt/output/result.pkl"
