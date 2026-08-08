# 8卡分布式训练，全量数据，50 epoch
# P0 v2: 修正LR schedule适配50ep + 温和时序权重 + 恢复梯度裁剪
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 80 \
    --epochs 50 \
    --extra_tag p0_v2
