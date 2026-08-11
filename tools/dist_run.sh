# 8卡分布式训练，全量数据，50 epoch
# P3: 修正cls/vel损失权重 + 固定随机种子 + 单卡eval
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 64 \
    --epochs 50 \
    --fix_random_seed \
    --workers 0 \
    --extra_tag p3_opt
