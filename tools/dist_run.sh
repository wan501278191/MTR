# 8卡分布式训练，全量数据，80 epoch
# 优化minADE/minFDE: 原始Waymo意图点 + 80ep + EMA + reg loss 1.5 + batch_size=64
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 64 \
    --epochs 80 \
    --fix_random_seed \
    --workers 0 \
    --extra_tag opt_adefde_v1
