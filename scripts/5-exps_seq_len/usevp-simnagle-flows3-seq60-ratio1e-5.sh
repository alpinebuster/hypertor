# 
# 1) TCUB:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-flows3-seq60-ratio1e-5.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/5-exps_seq_len/tcub-usevp-simnagle-flows3-seq60-ratio1e-5 > tcub-usevp-simnagle-flows3-seq60-ratio1e-5.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/5-exps_seq_len/usevp-simnagle-flows3-seq60-ratio1e-5.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/5-exps_seq_len/cicdarknet-usevp-simnagle-flows3-seq60-ratio1e-5 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-usevp-simnagle-flows3-seq60-ratio1e-5.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=0.00001 --num_flows=3 --tcp_seq_len=60 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=0 $@
