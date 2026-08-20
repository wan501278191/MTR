#!/usr/bin/env bash
# 赛事提交打包脚本
# 把镜像、文档、演示图片、模型、Dockerfile、result.pkl、代码打包压缩
# 在远程训练机 /root/wanqinghua/MTR 目录下执行
set -ex

MTR_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$MTR_DIR"

SUBMIT_DIR="/tmp/mtr_submission"
SUBMIT_TAR="/tmp/mtr_voyah_submission_pack.tar.gz"

echo "=== 1. 准备打包目录 ==="
rm -rf "$SUBMIT_DIR"
mkdir -p "$SUBMIT_DIR/演示图片" "$SUBMIT_DIR/model" "$SUBMIT_DIR/output" "$SUBMIT_DIR/MTR"

echo "=== 2. 复制文件 ==="

# 文档
echo "  [1/7] 复制文档..."
cp 算法说明文档.md 自测结果.md 提交说明文档.md "$SUBMIT_DIR/" 2>/dev/null || true
cp README.md "$SUBMIT_DIR/" 2>/dev/null || true

# Dockerfile
echo "  [2/7] 复制 Dockerfile..."
cp Dockerfile "$SUBMIT_DIR/"

# Docker 镜像 tar 包（必须存在，不存在则报错退出）
echo "  [3/7] 复制 Docker 镜像..."
if [ ! -f "mtr_voyah_submission.tar.gz" ]; then
    echo "错误: 镜像文件 mtr_voyah_submission.tar.gz 不存在，请先执行 build_docker.sh"
    exit 1
fi
cp -v mtr_voyah_submission.tar.gz "$SUBMIT_DIR/"

# 模型权重
echo "  [4/7] 复制模型权重..."
cp -v model/best_model.pth "$SUBMIT_DIR/model/"

# 测试结果
echo "  [5/7] 复制测试结果 result.pkl..."
mkdir -p "$SUBMIT_DIR/output"
cp -v test_output/result.pkl "$SUBMIT_DIR/output/"

# 演示图片（eval + test）
echo "  [6/7] 复制演示图片..."
cp -rv 演示图片/* "$SUBMIT_DIR/演示图片/" 2>/dev/null || echo "警告: 演示图片不存在"

# 代码（排除大文件）
echo "  [7/7] 复制代码..."
rsync -a --info=progress2 \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude 'swanlog/' \
    --exclude '*.tar' \
    --exclude '*.tar.gz' \
    --exclude '*.egg-info' \
    --exclude '*.so' \
    --exclude '.DS_Store' \
    ./ "$SUBMIT_DIR/MTR/"

echo "=== 3. 打包压缩 ==="
cd /tmp
rm -f "$SUBMIT_TAR"
# 显示打包进度：先计算总大小，再用 pv 监控；无 pv 则用 tar -v
TOTAL_SIZE=$(du -sb /tmp/mtr_submission | awk '{print $1}')
if command -v pv >/dev/null 2>&1; then
    tar -cf - -C /tmp mtr_submission | pv -s "$TOTAL_SIZE" | gzip > "$SUBMIT_TAR"
else
    echo "（安装 pv 可显示进度: apt-get install -y pv）"
    tar -czvf "$SUBMIT_TAR" -C /tmp mtr_submission
fi

echo "=== 4. 完成 ==="
ls -lh "$SUBMIT_TAR"
echo "提交包: $SUBMIT_TAR"
echo ""
echo "目录结构:"
find "$SUBMIT_DIR" -maxdepth 2 -type f -exec ls -lh {} \; | awk '{print $5, $NF}'
