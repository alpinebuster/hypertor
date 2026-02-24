# Call `./scripts/darkflow_1shot.sh` from the project root directory with following flags::
# 
#   --data_numpy_dir=<darkflow_cache>
#   --train_log_dir=<output_path>
# 
# e.g.
#    `nohup ./scripts/darkflow_1shot.sh --data_numpy_dir=./data/cicdarknet/cache --train_log_dir=./data/cicdarknet/logs/torch_1shot > darkflow_1shot.log 2>&1 &`
# 
python -m hypertor.train --num_layerwise_features=8 --default_num_channels=8 \
  --samples_transformer=7 --samples_cnn=35 --num_labels=7 --learning_rate=5e-4 \
  --train_steps=4000000 --steps_between_saves=1000 --lw_key_query_dim=1.0 \
  --lw_value_dim=1.0 --lw_inner_dim=1.0 --cnn_model_name='maxpool-4-layer' \
  --embedding_dim=32 --num_layers=3 --stride=1 --heads=2 \
  --shared_feature_extractor='3-layer-bn' --shared_input_dim=15 --shared_features_dim=32 \
  --shared_feature_extractor_padding=same --layerwise_generator=joint \
  --nolw_use_nonlinear_feature --lw_weight_allocation=output --nolw_generate_bias \
  --nolw_generate_bn --nouse_decoder --noadd_trainable_weights \
  --balanced_batches --per_label_augmentation \
  --smooth_probability=0.0 --negate_probability=0.0 \
  --train_dataset='cicdarknet:0-6' \
  --test_dataset='cicdarknet:0-6' --eval_datasets='cicdarknet' \
  --num_task_evals=1024 --num_eval_batches=16 --eval_batch_size=70 \
  --shuffle_labels_seed=2026 \
  --test_smooth_probability=0.0 --test_negate_probability=0.0 \
  --warmup_steps=0 --use_variable_proto --number_of_trained_cnn_layers=0 \
  --push_start=7500 --push_warmup_steps=3000 --push_last_steps=2000 \
  --use_darkflow --flows_shape=15,30 \
  --darknet_ratio=0.001 --num_pkt_features=3 --num_flows=1 --tcp_seq_len=30 \
  --simulate_nagle --nosimulate_packet_shift --nosimulate_packet_loss \
  --simulation_seeds=2000,2026 --simulation_samples_perclass=6 \
  --gpus=0 $@
