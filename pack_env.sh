#!/usr/bin/env bash
# 在远程训练机上打包 conda 环境（离线用）
# 用法: bash pack_env.sh
set -ex

SITE_PACKAGES="/data/miniforge3/envs/mtr/lib/python3.8/site-packages"
CONDA_ENV="/data/miniforge3/envs/mtr"

# 安装 conda-pack（如果没装）
pip install conda-pack 2>/dev/null || conda install -y -c conda-forge conda-pack

# === 彻底清除 editable 安装的所有痕迹（conda-pack 不支持 editable 包）===
pip uninstall -y mtr 2>/dev/null || true
rm -f "$SITE_PACKAGES"/*.egg-link 2>/dev/null || true
if [ -f "$SITE_PACKAGES/easy-install.pth" ]; then
    grep -v "MTR\|mtr" "$SITE_PACKAGES/easy-install.pth" > /tmp/easy-install.pth || true
    cp /tmp/easy-install.pth "$SITE_PACKAGES/easy-install.pth"
    [ ! -s "$SITE_PACKAGES/easy-install.pth" ] && rm -f "$SITE_PACKAGES/easy-install.pth" || true
fi
rm -rf "$SITE_PACKAGES"/mtr*.egg* 2>/dev/null || true
rm -rf /root/wanqinghua/MTR/*.egg-info 2>/dev/null || true

# === 清理 conda 环境内冗余文件，大幅减小打包体积 ===
echo "=== 清理 conda 环境缓存和冗余文件 ==="

# conda 包缓存
conda clean --all -y 2>/dev/null || true

# pip 缓存
pip cache purge 2>/dev/null || true
rm -rf "$CONDA_ENV/.cache" 2>/dev/null || true
rm -rf "$CONDA_ENV/pkgs" 2>/dev/null || true

# __pycache__ 和 .pyc
find "$CONDA_ENV" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$CONDA_ENV" -name '*.pyc' -delete 2>/dev/null || true
find "$CONDA_ENV" -name '*.pyo' -delete 2>/dev/null || true

# 测试文件和文档
find "$SITE_PACKAGES" -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -name '*.so' -name '*dSYM*' -exec rm -rf {} + 2>/dev/null || true

# aotriton/shader caches
rm -rf "$CONDA_ENV/lib/python3.8/site-packages/torchinductor" 2>/dev/null || true
rm -rf "$CONDA_ENV/lib/python3.8/site-packages/triton" 2>/dev/null || true

echo "清理后环境大小:"
du -sh "$CONDA_ENV" 2>/dev/null || true

# 验证无残留
echo "=== 检查 editable 残留 ==="
find "$SITE_PACKAGES" -name "*.egg-link" 2>/dev/null || echo "无 egg-link"
cat "$SITE_PACKAGES/easy-install.pth" 2>/dev/null || echo "无 easy-install.pth"

# 打包 mtr 环境
conda pack -n mtr -o mtr.tar.gz --force

# === 恢复训练机的 mtr 可 import（不需要重新编译 CUDA 算子）===
# 用 .pth 文件代替 setup.py develop，.so 文件已存在无需重编译
echo "/root/wanqinghua/MTR" > "$SITE_PACKAGES/mtr_path.pth"
echo "已创建 mtr_path.pth，验证 import:"
python -c "import mtr; print('import mtr OK')" 2>/dev/null || echo "警告: import mtr 失败，请手动设置 PYTHONPATH=/root/wanqinghua/MTR"

echo "=== 完成 ==="
ls -lh mtr.tar.gz
