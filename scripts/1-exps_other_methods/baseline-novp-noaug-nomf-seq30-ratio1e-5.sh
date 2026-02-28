# 
# 1) TCUB:
#    `nohup ./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1e-5.sh --data_numpy_dir=./data/tcub/cache --train_log_dir=./data/tcub/logs/baseline-novp-noaug-nomf-seq30-ratio1e-5 > tcub-baseline-novp-noaug-nomf-seq30-ratio1e-5.log 2>&1 &`
# 
# 2) CICDARKNET:
#    `nohup ./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1e-5.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/baseline-novp-noaug-nomf-seq30-ratio1e-5 --train_dataset='cicdarknet:0-6' --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' > cicdarknet-baseline-novp-noaug-nomf-seq30-ratio1e-5.log 2>&1 &`
# 
./scripts/1-exps_other_methods/baseline-novp-noaug-nomf-seq30-ratio1.sh \
  --darknet_ratio=0.00001 \
  --gpus=0 $@
