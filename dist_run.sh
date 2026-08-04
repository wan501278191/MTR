# 8卡分布式训练，全量数据，30 epoch
bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/waymo/mtr_voyah_data.yaml \
    --batch_size 80 \
    --epochs 30 \
    --extra_tag baseline
