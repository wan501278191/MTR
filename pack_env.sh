#!/usr/bin/env bash
# 在远程训练机上打包 conda 环境（离线用）
# 用法: bash pack_env.sh
set -ex

SITE_PACKAGES="/data/miniforge3/envs/mtr/lib/python3.8/site-packages"
CONDA_ENV="/data/miniforge3/envs/mtr"

# 安装 conda-pack（如果没装）
pip install conda-pack 2>/dev/null || conda install -y -c conda-forge conda-pack

# === 彻底清除所有 editable 安装痕迹 ===
# conda-pack 不支持 editable 包，需删除所有形式：
#   1. egg-link 方式: *.egg-link + easy-install.pth
#   2. pip 新版方式: __editable__.*.pth + __editable___*_finder.py
#   3. 自定义 .pth: mtr_path.pth 等
echo "=== 清除 editable 安装痕迹 ==="

pip uninstall -y mtr MotionTransformer 2>/dev/null || true

# 删除 egg-link 方式
rm -f "$SITE_PACKAGES"/*.egg-link 2>/dev/null || true
if [ -f "$SITE_PACKAGES/easy-install.pth" ]; then
    grep -v "MTR\|mtr\|MotionTransformer" "$SITE_PACKAGES/easy-install.pth" > /tmp/easy-install.pth || true
    cp /tmp/easy-install.pth "$SITE_PACKAGES/easy-install.pth"
    [ ! -s "$SITE_PACKAGES/easy-install.pth" ] && rm -f "$SITE_PACKAGES/easy-install.pth" || true
fi

# 删除 pip 新版 editable 方式 (__editable__)
rm -f "$SITE_PACKAGES"/__editable__* 2>/dev/null || true
rm -f "$SITE_PACKAGES"/__editable___* 2>/dev/null || true

# 删除自定义 mtr_path.pth（打包前不能有指向本地路径的 .pth）
rm -f "$SITE_PACKAGES/mtr_path.pth" 2>/dev/null || true

# 删除 egg-info
rm -rf "$SITE_PACKAGES"/mtr*.egg* 2>/dev/null || true
rm -rf "$SITE_PACKAGES"/MotionTransformer*.egg* 2>/dev/null || true
rm -rf /root/wanqinghua/MTR/*.egg-info 2>/dev/null || true
rm -rf /root/wanqinghua/MTR/MotionTransformer.egg-info 2>/dev/null || true

# === 验证无 editable 残留 ===
echo "=== 检查 editable 残留 ==="
echo "--- egg-link ---"
find "$SITE_PACKAGES" -name "*.egg-link" 2>/dev/null || echo "无 egg-link"
echo "--- __editable__ ---"
find "$SITE_PACKAGES" -name "__editable__*" 2>/dev/null || echo "无 __editable__"
echo "--- easy-install.pth ---"
cat "$SITE_PACKAGES/easy-install.pth" 2>/dev/null || echo "无 easy-install.pth"
echo "--- mtr_path.pth ---"
cat "$SITE_PACKAGES/mtr_path.pth" 2>/dev/null || echo "无 mtr_path.pth"

# === 清理缓存（仅清理缓存，不删除 conda 管理的 tests/__pycache__）===
echo "=== 清理 conda 环境缓存 ==="

# conda 包缓存
conda clean --all -y 2>/dev/null || true

# pip 缓存
pip cache purge 2>/dev/null || true
rm -rf "$CONDA_ENV/.cache" 2>/dev/null || true
rm -rf "$CONDA_ENV/pkgs" 2>/dev/null || true

# 仅清理第三方包的 __pycache__（跳过 conda 管理的标准库和 setuptools）
# 保留 __pycache__（TF 等包的 circular import 依赖编译缓存）
# find "$SITE_PACKAGES" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
# find "$SITE_PACKAGES" -name '*.pyc' -delete 2>/dev/null || true

# aotriton/shader caches（推理不需要）
rm -rf "$SITE_PACKAGES/torchinductor" 2>/dev/null || true
rm -rf "$SITE_PACKAGES/triton" 2>/dev/null || true

echo "清理后环境大小:"
du -sh "$CONDA_ENV" 2>/dev/null || true

# 打包 mtr 环境
# --ignore-missing-files: 允许 __pycache__ 被删除后仍能打包
echo "=== 打包 conda 环境 ==="
conda pack -n mtr -o mtr.tar.gz --force --ignore-missing-files

# === 恢复训练机的 mtr 可 import（不需要重新编译 CUDA 算子）===
# 用 .pth 文件代替 setup.py develop，.so 文件已存在无需重编译
echo "/root/wanqinghua/MTR" > "$SITE_PACKAGES/mtr_path.pth"
echo "已创建 mtr_path.pth，验证 import:"
python -c "import mtr; print('import mtr OK')" 2>/dev/null || echo "警告: import mtr 失败，请手动设置 PYTHONPATH=/root/wanqinghua/MTR"

echo "=== 完成 ==="
ls -lh mtr.tar.gz
