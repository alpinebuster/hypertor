# 
# 1) TCUB:
#    `nohup ./scripts/2-exps_tcp_augmentation/novp-simshift-nomf-seq30-ratio1.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/2-exps_tcp_augmentation/tcub-novp-simshift-nomf-seq30-ratio1 > tcub-novp-simshift-nomf-seq30-ratio1.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/2-exps_tcp_augmentation/novp-simshift-nomf-seq30-ratio1.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-novp-simshift-nomf-seq30-ratio1 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-novp-simshift-nomf-seq30-ratio1.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --darknet_ratio=1 \
  --nosimulate_nagle --simulate_packet_shift --nosimulate_packet_loss \
  --gpus=1 $@
