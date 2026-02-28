# 
# 1) TCUB:
#    `nohup ./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1e-3.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/1-exps_other_methods/tcub-baseline-novp-noaug-nomf-seq30-ratio1e-3 > tcub-baseline-novp-noaug-nomf-seq30-ratio1e-3.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1e-3.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/1-exps_other_methods/cicdarknet-baseline-novp-noaug-nomf-seq30-ratio1e-3 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-baseline-novp-noaug-nomf-seq30-ratio1e-3.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --darknet_ratio=0.001 \
  --gpus=0 $@
