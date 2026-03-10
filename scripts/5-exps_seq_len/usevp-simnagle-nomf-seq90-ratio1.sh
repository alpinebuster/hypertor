# 
# 1) TCUB:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-nomf-seq90-ratio1.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/5-exps_seq_len/tcub-usevp-simnagle-nomf-seq90-ratio1 > tcub-usevp-simnagle-nomf-seq90-ratio1.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-nomf-seq90-ratio1.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/5-exps_seq_len/cicdarknet-usevp-simnagle-nomf-seq90-ratio1 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-usevp-simnagle-nomf-seq90-ratio1.log 2>&1 &`
# 
# 3) SafeSurf2025:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-nomf-seq90-ratio1.sh --data_numpy_dir=./data/safesurf2025/cache --train_log_dir=./data/safesurf2025/logs/5-exps_seq_len/safesurf2025-usevp-simnagle-nomf-seq90-ratio1 --samples_transformer=30 --samples_cnn=42 --num_labels=6 --train_dataset='safesurf2025:0-5' --test_dataset='safesurf2025:0-5' --eval_datasets='safesurf2025' --learning_rate=5e-3 > safesurf2025-usevp-simnagle-nomf-seq90-ratio1.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=1 --num_flows=1 --tcp_seq_len=90 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=1 $@
