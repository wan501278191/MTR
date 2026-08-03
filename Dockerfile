# MTR 轨迹预测提交镜像
# 赛事要求: Ubuntu 20.04/22.04, CUDA < 12.3
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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

# 拷贝代码
COPY . /workspace/MTR/

# 编译 CUDA 算子
RUN python3 setup.py develop

# 数据和模型通过 volume 挂载，不入镜像
# docker run -v /host/data:/workspace/data \
#            -v /host/model:/workspace/model \
#            -v /host/output:/workspace/output \
#            mtr-voyah-submission

ENV DATA_ROOT=/workspace/data
ENV CKPT_PATH=/workspace/model/checkpoint_epoch_50.pth
ENV OUTPUT_DIR=/workspace/output

# 入口：推理并生成 result.pkl
CMD ["python3", "tools/test.py", \
     "--cfg_file", "cfgs/waymo/mtr_voyah_data.yaml", \
     "--ckpt", "/workspace/model/checkpoint_epoch_50.pth", \
     "--extra_tag", "submission", \
     "--batch_size", "80", \
     "--workers", "8", \
     "--save_to_file", \
     "--set", "DATA_CONFIG.DATA_ROOT", "/workspace/data", \
           "DATA_CONFIG.SPLIT_DIR.test", "processed_scenarios_testing_B1_part", \
           "DATA_CONFIG.INFO_FILE.test", "processed_scenarios_testB1_part_infos.pkl"]
