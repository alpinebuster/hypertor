# Call from the project root directory:
#    `nohup ./scripts/darkflow_aug.sh > darkflow_aug.log 2>&1 &`
# 
python -m hypertor.aug.darkflow \
  --simulate_nagle --simulate_packet_shift \
  --simulate_packet_loss --simulate_packet_duplication \
  --tcp_seq_len=30 $@
