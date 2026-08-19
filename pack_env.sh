#!/usr/bin/env bash
# 在远程训练机上打包 conda 环境（离线用）
# 用法: bash pack_env.sh
set -ex

# 安装 conda-pack（如果没装）
pip install conda-pack 2>/dev/null || conda install -y -c conda-forge conda-pack

# 打包 mtr 环境
conda pack -n mtr -o mtr.tar.gz --force

echo "=== 完成 ==="
ls -lh mtr.tar.gz
echo "将 mtr.tar.gz 放到 Dockerfile 同级目录后执行: bash build_docker.sh"
