# 
# 1) TCUB:
#    `nohup ./scripts/3-exps_variable_prototypes/usevp-noaug-nomf-seq30-ratio1.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/3-exps_variable_prototypes/tcub-usevp-noaug-nomf-seq30-ratio1 > tcub-usevp-noaug-nomf-seq30-ratio1.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/3-exps_variable_prototypes/usevp-noaug-nomf-seq30-ratio1.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-usevp-noaug-nomf-seq30-ratio1.log 2>&1 &`
# 
# 3) SafeSurf2025:
#    `nohup ./scripts/3-exps_variable_prototypes/usevp-noaug-nomf-seq30-ratio1.sh --data_numpy_dir=./data/safesurf2025/cache --train_log_dir=./data/safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-usevp-noaug-nomf-seq30-ratio1 --samples_transformer=30 --samples_cnn=42 --num_labels=6 --train_dataset='safesurf2025:0-5' --test_dataset='safesurf2025:0-5' --eval_datasets='safesurf2025' > safesurf2025-usevp-noaug-nomf-seq30-ratio1.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=1 \
  --nosimulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=0 $@
