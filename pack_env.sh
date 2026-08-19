#!/usr/bin/env bash
# 在远程训练机上打包 conda 环境（离线用）
# 用法: bash pack_env.sh
set -ex

SITE_PACKAGES="/data/miniforge3/envs/mtr/lib/python3.8/site-packages"

# 安装 conda-pack（如果没装）
pip install conda-pack 2>/dev/null || conda install -y -c conda-forge conda-pack

# === 彻底清除 editable 安装的所有痕迹 ===
# 1. 卸载 mtr 包
pip uninstall -y mtr 2>/dev/null || true

# 2. 删除 egg-link 文件
rm -f "$SITE_PACKAGES"/*.egg-link 2>/dev/null || true

# 3. 从 easy-install.pth 中删除 MTR 相关行
if [ -f "$SITE_PACKAGES/easy-install.pth" ]; then
    grep -v "MTR\|mtr" "$SITE_PACKAGES/easy-install.pth" > /tmp/easy-install.pth || true
    cp /tmp/easy-install.pth "$SITE_PACKAGES/easy-install.pth"
    # 如果 pth 文件空了就删掉
    [ ! -s "$SITE_PACKAGES/easy-install.pth" ] && rm -f "$SITE_PACKAGES/easy-install.pth" || true
fi

# 4. 删除 mtr 的 egg-info
rm -rf "$SITE_PACKAGES"/mtr*.egg* 2>/dev/null || true
rm -rf /root/wanqinghua/MTR/*.egg-info 2>/dev/null || true

# 5. 验证没有 editable 残留
echo "=== 检查 editable 残留 ==="
find "$SITE_PACKAGES" -name "*.egg-link" 2>/dev/null || echo "无 egg-link"
cat "$SITE_PACKAGES/easy-install.pth" 2>/dev/null || echo "无 easy-install.pth"

# 打包 mtr 环境
conda pack -n mtr -o mtr.tar.gz --force

# 打包后重新安装 editable 包（不影响训练机后续使用）
cd /root/wanqinghua/MTR
pip install -e . --no-deps 2>/dev/null || python setup.py develop --no-deps 2>/dev/null || true

echo "=== 完成 ==="
ls -lh mtr.tar.gz
echo "将 mtr.tar.gz 放到 Dockerfile 同级目录后执行: bash build_docker.sh"
