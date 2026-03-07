# 
# 1) TCUB:
#    `nohup ./scripts/2-exps_tcp_augmentation/usevp-simall-nomf-seq30-ratio1e-3.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simall-nomf-seq30-ratio1e-3 > tcub-usevp-simall-nomf-seq30-ratio1e-3.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/2-exps_tcp_augmentation/usevp-simall-nomf-seq30-ratio1e-3.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simall-nomf-seq30-ratio1e-3 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-usevp-simall-nomf-seq30-ratio1e-3.log 2>&1 &`
# 
# 3) SafeSurf2025:
#    `nohup ./scripts/2-exps_tcp_augmentation/usevp-simall-nomf-seq30-ratio1e-3.sh --data_numpy_dir=./data/safesurf2025/cache --train_log_dir=./data/safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simall-nomf-seq30-ratio1e-3 --num_labels=6 --train_dataset='safesurf2025:0-5' --test_dataset='safesurf2025:0-5' --eval_datasets='safesurf2025' > safesurf2025-usevp-simall-nomf-seq30-ratio1e-3.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=0.001 \
  --simulate_nagle --simulate_packet_shift --simulate_packet_loss \
  --gpus=0 $@
