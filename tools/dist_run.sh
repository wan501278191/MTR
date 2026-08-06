# 8卡分布式训练，全量数据，50 epoch
# P0+P1优化版（可学习意图查询 + 运动学注入 + GRU时序编码 + 软分配 + 机动辅助头）
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 64 \
    --epochs 50 \
    --extra_tag opt_v2
