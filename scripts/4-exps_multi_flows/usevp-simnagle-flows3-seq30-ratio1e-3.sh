# 
# 1) TCUB:
#    `nohup ./scripts/4-exps_multi_flows/usevp-simnagle-flows3-seq30-ratio1e-3.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows3-seq30-ratio1e-3 > tcub-usevp-simnagle-flows3-seq30-ratio1e-3.log 2>&1 &`
# 
# 2) CICDARKNET: NO enough FP data
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=0.001 --num_flows=3 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=0 $@
