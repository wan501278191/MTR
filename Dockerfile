# MTR 轨迹预测提交镜像（含模型权重）
# 赛事要求: Ubuntu 22.04, CUDA < 12.3
# 多阶段构建：stage-1 解压+清理 conda 环境，stage-2 最终运行镜像

# ===== Stage 1: 解压并清理 conda 环境 =====
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS env-builder

ENV MTR_ENV=/opt/conda/envs/mtr

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates findutils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "$MTR_ENV"
ADD mtr.tar.gz /opt/conda/envs/mtr/

# 清理解压后的环境
RUN find "$MTR_ENV" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    find "$MTR_ENV" -name '*.pyc' -delete 2>/dev/null; \
    find "$MTR_ENV" -name '*.pyo' -delete 2>/dev/null; \
    rm -rf "$MTR_ENV/pkgs" 2>/dev/null; \
    rm -rf "$MTR_ENV/.cache" 2>/dev/null; \
    find "$MTR_ENV/lib/python3.8/site-packages" -type d -name 'tests' -exec rm -rf {} + 2>/dev/null; \
    rm -rf "$MTR_ENV/lib/python3.8/site-packages/triton" 2>/dev/null; \
    rm -rf "$MTR_ENV/lib/python3.8/site-packages/torchinductor" 2>/dev/null; \
    true

# ===== Stage 2: 最终运行镜像 =====
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    MTR_ENV=/opt/conda/envs/mtr

# 仅安装运行时依赖（不含 build-essential / ninja-build 等开发工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 从 stage-1 复制已清理的 conda 环境
COPY --from=env-builder "$MTR_ENV" "$MTR_ENV"
RUN "$MTR_ENV/bin/python" "$MTR_ENV/bin/conda-unpack"

WORKDIR /workspace

# 拷贝算法代码
COPY . /workspace/MTR

# 内置模型权重
RUN mkdir -p /workspace/model
COPY model/best_model.pth /workspace/model/best_model.pth

# 环境变量
ENV PATH=$MTR_ENV/bin:$PATH \
    PYTHONPATH=/workspace/MTR

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace/MTR

# 挂载约定:
#   /mnt/data          — 数据集根目录
#   /mnt/output        — 推理结果输出目录
# 容器启动即执行推理，生成 result.pkl 并拷贝到 /mnt/output
COPY docker_entrypoint.sh /workspace/docker_entrypoint.sh
RUN chmod +x /workspace/docker_entrypoint.sh

CMD ["/workspace/docker_entrypoint.sh"]
