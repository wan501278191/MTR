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
RUN TF_INT="$MTR_ENV/lib/python3.8/site-packages/tensorflow/_api/v2/__internal__"; \
    if [ ! -d "$TF_INT/test" ]; then \
        mkdir -p "$TF_INT/test" && \
        echo "from tensorflow.python.util import module_wrapper as _module_wrapper; test = _module_wrapper.module_wrapper(__name__)" > "$TF_INT/test/__init__.py" && \
        echo "已修复 TF test 模块"; \
    fi; true

WORKDIR /workspace

# 拷贝算法代码
COPY . /workspace/MTR

# 删除 COPY . 带入的大文件（ADD 阶段已用，不需要保留在最终镜像）
RUN rm -rf /workspace/MTR/mtr.tar.gz \
    /workspace/MTR/mtr_voyah_submission.tar.gz \
    /workspace/MTR/*.tar.gz \
    /workspace/MTR/output /workspace/MTR/test_output \
    /workspace/MTR/swanlog /workspace/MTR/build \
    /workspace/MTR/MotionTransformer.egg-info 2>/dev/null; true

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
