# MTR 轨迹预测提交镜像（含模型权重）
# 赛事要求: Ubuntu 22.04, CUDA < 12.3
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3-dev \
    git build-essential ninja-build \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MTR

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt /workspace/MTR/requirements.txt
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir \
        waymo-open-dataset-tf-2-6-0 \
        tensorflow==2.6.0

# 拷贝全部代码
COPY . /workspace/MTR/

# 编译 CUDA 算子
RUN python3 setup.py develop

# 内置 EMA 最佳权重（验证集 mAP 最优；满足"镜像内置权重"要求）
RUN mkdir -p /workspace/model
COPY model/best_model_ema.pth /workspace/model/best_model_ema.pth

# 环境变量（容器内默认路径）
ENV DATA_ROOT=/workspace/data
ENV CKPT_PATH=/workspace/model/best_model_ema.pth
ENV OUTPUT_DIR=/workspace/output

# 入口：推理并生成 result.pkl（精度评测与速度评测使用同一权重）
CMD ["python3", "tools/test.py", \
     "--cfg_file", "cfgs/waymo/mtr_voyah_data.yaml", \
     "--ckpt", "/workspace/model/best_model_ema.pth", \
     "--extra_tag", "submission", \
     "--batch_size", "80", \
     "--workers", "8", \
     "--save_to_file", \
     "--set", "DATA_CONFIG.DATA_ROOT", "/workspace/data", \
           "DATA_CONFIG.SPLIT_DIR.test", "processed_scenarios_testing_B1_part", \
           "DATA_CONFIG.INFO_FILE.test", "processed_scenarios_testB1_part_infos.pkl"]
