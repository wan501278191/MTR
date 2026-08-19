#!/usr/bin/env bash
# 在远程训练机上打包 conda 环境（离线用）
# 用法: bash pack_env.sh
set -ex

# 安装 conda-pack（如果没装）
pip install conda-pack 2>/dev/null || conda install -y -c conda-forge conda-pack

# conda-pack 不支持 editable 包，临时卸载 MTR 的 editable 安装
pip uninstall -y mtr 2>/dev/null || true
# 清理 egg-link 残留
find /data/miniforge3/envs/mtr/lib/python3.8/site-packages -name "*.egg-link" -delete 2>/dev/null || true
find /data/miniforge3/envs/mtr/lib/python3.8/site-packages -name "mtr*.egg*" -delete 2>/dev/null || true

# 打包 mtr 环境
conda pack -n mtr -o mtr.tar.gz --force

# 打包后重新安装 editable 包（不影响训练机后续使用）
cd /root/wanqinghua/MTR
pip install -e . --no-deps 2>/dev/null || python setup.py develop --no-deps 2>/dev/null || true

echo "=== 完成 ==="
ls -lh mtr.tar.gz
echo "将 mtr.tar.gz 放到 Dockerfile 同级目录后执行: bash build_docker.sh"
