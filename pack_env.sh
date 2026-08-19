#!/usr/bin/env bash
# 在远程训练机上打包 conda 环境（离线用）
# 用法: bash pack_env.sh
set -ex

SITE_PACKAGES="/data/miniforge3/envs/mtr/lib/python3.8/site-packages"

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
