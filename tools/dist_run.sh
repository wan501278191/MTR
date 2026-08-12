# 8卡分布式训练，全量数据，50 epoch
# P5+P6: 百分位裁剪重新聚类意图点 + 修正LR schedule
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 64 \
    --epochs 50 \
    --fix_random_seed \
    --workers 0 \
    --extra_tag p5_pct90_lrsched
