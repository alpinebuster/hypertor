# 
# 1) TCUB:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-flows3-seq90-ratio1e-5.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/5-exps_seq_len/tcub-usevp-simnagle-flows3-seq90-ratio1e-5 > tcub-usevp-simnagle-flows3-seq90-ratio1e-5.log 2>&1 &`
# 
# 2) CICDARKNET: NO enough FP data
# 
# 3) SafeSurf2025:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-flows3-seq90-ratio1e-5.sh --data_numpy_dir=./data/safesurf2025/cache --train_log_dir=./data/safesurf2025/logs/5-exps_seq_len/safesurf2025-usevp-simnagle-flows3-seq90-ratio1e-5 --samples_transformer=30 --samples_cnn=42 --num_labels=6 --train_dataset='safesurf2025:0-5' --test_dataset='safesurf2025:0-5' --eval_datasets='safesurf2025' > safesurf2025-usevp-simnagle-flows3-seq90-ratio1e-5.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=0.00001 --num_flows=3 --tcp_seq_len=90 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=1 $@
