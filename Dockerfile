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
# 单独 COPY mtr.tar.gz（.dockerignore 已排除 *.tar.gz，但显式 COPY 仍可用）
COPY mtr.tar.gz /tmp/mtr.tar.gz
RUN tar xzf /tmp/mtr.tar.gz -C "$MTR_ENV" && rm /tmp/mtr.tar.gz

# 清理解压后的环境（保留 __pycache__，TF 依赖编译缓存）
RUN rm -rf "$MTR_ENV/pkgs" 2>/dev/null; \
    rm -rf "$MTR_ENV/.cache" 2>/dev/null; \
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

# 修复 conda-pack 可能导致的 TF 缺失模块（tensorflow._api.v2.__internal__.test）
# 始终覆盖为空 __init__.py，确保没有残留的错误内容
RUN TF_INT="$MTR_ENV/lib/python3.8/site-packages/tensorflow/_api/v2/__internal__"; \
    mkdir -p "$TF_INT/test" && \
    touch "$TF_INT/test/__init__.py" && \
    echo "已修复 TF test 模块"; true

WORKDIR /workspace

# 拷贝算法代码（显式列出目录，避免带入 mtr.tar.gz 等大文件）
COPY mtr/ /workspace/MTR/mtr/
COPY tools/ /workspace/MTR/tools/
COPY data/ /workspace/MTR/data/
COPY docs/ /workspace/MTR/docs/
COPY setup.py requirements.txt LICENSE /workspace/MTR/

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
