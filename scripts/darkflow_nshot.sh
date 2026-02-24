# Call from the project root directory:
#    `./scripts/darkflow_nshot.sh` with flags
#    "--data_numpy_dir=<darkflow_cache> --train_log_dir=<output_path>"
# 
# e.g.
#    `nohup ./scripts/darkflow_nshot.sh --data_numpy_dir=./data/darkflow/cache --train_log_dir=./data/darkflow/logs/torch_nshot > darkflow_nshot.log 2>&1 &`
# 
./scripts/darkflow_1shot.sh --samples_transformer=100 --samples_cnn=100 $@
