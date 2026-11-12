#
#    `nohup ./scripts/eval-darkflow_nshot.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/torch_nshot > eval-darkflow_cicdarknet_nshot.log 2>&1 &`
# 
./scripts/eval-darkflow_1shot.sh --samples_transformer=35 --samples_cnn=70 \
  --train_dataset='cicdarknet:0-6' \
  --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' \
  --use_variable_proto \
  --use_darkflow --flows_shape=3,30 --shared_input_dim=3 --num_pkt_features=3 \
  --darknet_ratio=0.001 --num_flows=1 --tcp_seq_len=30 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --simulation_seeds=2000,2026 --simulation_samples_perclass=6 \
  --gpus=0 $@
