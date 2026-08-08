# 8卡分布式训练，全量数据，60 epoch
# P0优化: LR schedule适配60epoch + 时序损失权重 + batch_size统一 + 梯度裁剪收紧
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 80 \
    --epochs 60 \
    --extra_tag p0_opt
