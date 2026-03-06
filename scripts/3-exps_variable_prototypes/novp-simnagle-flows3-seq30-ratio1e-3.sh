# 
# 1) TCUB:
#    `nohup ./scripts/3-exps_variable_prototypes/novp-simnagle-flows3-seq30-ratio1e-3.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/3-exps_variable_prototypes/tcub-novp-simnagle-flows3-seq30-ratio1e-3 > tcub-novp-simnagle-flows3-seq30-ratio1e-3.log 2>&1 &`
# 
# 2) CICDARKNET: NO enough FP data
#
# 3) SafeSurf2025:
#    `nohup ./scripts/3-exps_variable_prototypes/novp-simnagle-flows3-seq30-ratio1e-3.sh --data_numpy_dir=./data/safesurf2025/cache --train_log_dir=./data/safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-novp-simnagle-flows3-seq30-ratio1e-3 --train_dataset='safesurf2025:0-5' --test_dataset='safesurf2025:0-5' --eval_datasets='safesurf2025' > safesurf2025-novp-simnagle-flows3-seq30-ratio1e-3.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --nouse_variable_proto \
  --darknet_ratio=0.001 --num_flows=3 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --gpus=1 $@
