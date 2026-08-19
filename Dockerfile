# MTR 轨迹预测提交镜像（含模型权重）
# 赛事要求: Ubuntu 22.04, CUDA < 12.3
# 用 conda-pack 打包完整环境，构建时无需网络
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    MTR_ENV=/opt/conda/envs/mtr

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 \
    tzdata build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 解压 conda-pack 打包的完整 Python 环境
RUN mkdir -p "$MTR_ENV"
ADD mtr.tar.gz /opt/conda/envs/mtr/
RUN "$MTR_ENV/bin/python" "$MTR_ENV/bin/conda-unpack"

# 拷贝算法代码
COPY . /workspace/MTR

# 内置模型权重
RUN mkdir -p /workspace/model
COPY model/best_model_ema.pth /workspace/model/best_model_ema.pth

# 内置参考输出结果（随镜像提供）
COPY output/result.pkl /workspace/output/result.pkl

# 环境变量
ENV PATH=$MTR_ENV/bin:$PATH \
    PYTHONPATH=/workspace/MTR

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace/MTR

# 挂载约定:
#   /mnt/data          — 数据集根目录
#   /mnt/output        — 推理结果输出目录
# 容器启动即执行推理，生成 result.pkl
CMD ["python", "tools/test.py", \
     "--cfg_file", "cfgs/waymo/mtr_voyah_data.yaml", \
     "--ckpt", "/workspace/model/best_model_ema.pth", \
     "--extra_tag", "submission", \
     "--batch_size", "32", \
     "--workers", "8", \
     "--save_to_file", \
     "--set", "DATA_CONFIG.DATA_ROOT", "/mnt/data", \
           "DATA_CONFIG.SPLIT_DIR.test", "processed_scenarios_testing_B1_part", \
           "DATA_CONFIG.INFO_FILE.test", "processed_scenarios_testB1_part_infos.pkl"]
