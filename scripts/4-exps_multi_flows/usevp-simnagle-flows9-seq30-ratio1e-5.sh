# 
# 1) TCUB:
#    `nohup ./scripts/3-exps_variable_prototypes/usevp-simnagle-flows9-seq30-ratio1e-5.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/3-exps_variable_prototypes/tcub-usevp-simnagle-flows9-seq30-ratio1e-5 > tcub-usevp-simnagle-flows9-seq30-ratio1e-5.log 2>&1 &`
# 
# 2) CICDARKNET: NO enough FP data
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --use_variable_proto \
  --darknet_ratio=0.00001 --num_flows=9 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=0 $@
