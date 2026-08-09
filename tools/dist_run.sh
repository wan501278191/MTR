# 8卡分布式训练，全量数据，50 epoch
# P1: 修正翻转增强(跳过VEHICLE场景) + CYCLIST场景过采样2x
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 80 \
    --epochs 50 \
    --extra_tag p1_opt
