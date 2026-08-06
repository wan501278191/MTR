# 8卡分布式训练，全量数据，50 epoch
# P0+P1优化版（可学习意图查询 + 运动学注入 + GRU时序编码 + 软分配 + 机动辅助头）

# 清理残留 GPU 进程，避免 OOM
echo "========== 清理残留进程 =========="
pkill -9 -f "train.py" 2>/dev/null || true
sleep 2

echo "========== 检查 GPU 状态 =========="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "========== 开始训练 (opt_v2, 50 epoch, batch_size=64) =========="
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 64 \
    --epochs 50 \
    --extra_tag opt_v2
