# Darkflow Dataset

The Vec is converted to string representation in csv file.

## CSV Format

```txt
"pkt_dir_seq": Vec<i32>, // Direction (/ -)
"pkt_len_seq": Vec<i32>,
"pkt_usec_seq": Vec<i64>, // Relative time us

// ===== TCP flags =====
"tcp_window_seq": Vec<u16>,

"fin_seq": Vec<u8>,
"syn_seq": Vec<u8>,
"rst_seq": Vec<u8>,
"psh_seq": Vec<u8>,
"ack_seq": Vec<u8>,
"urg_seq": Vec<u8>, // Rarely used
"cwr_seq": Vec<u8>, // Rarely used
"ece_seq": Vec<u8>, // Rarely used

// ===== IP =====
"ttl_seq": Vec<u8>,
"ip_df_seq": Vec<u8>,
"ip_mf_seq": Vec<u8>,
```

### Classes

Perform a two-stage classification: first determine if the flow is Tor, then classify it into one of seven traffic categories.

- Browsing
- Email
- Chat
- Streaming (Audio/Video)
- FTP
- VoIP
- P2P


## Getting started

Run from `hypertor` project's root dir

```sh
cd ./hypertor/aug/

nohup python ./fp_multi_flows.py --dataset tcub --ratio 0.00001 --maxrate 0.365 --seed 2026 > fp_multi_flows_tcub_maxrate0.365_ratio1e-05_seed_2026.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset all --ratio 0.00001 --maxrate 0.365 --seed "2015-2026" > fp_multi_flows_maxrate0.365_ratio1e-05.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset all --ratio 1 --maxrate 0.165 --seed "2000-2014" > fp_multi_flows_maxrate0.165_ratio1.0.log 2>&1 & 

nohup python ./fp_multi_flows.py --dataset safesurf2025 --ratio 1 --maxrate 0.165 --seed "2000-2014" > fp_multi_flows_safesurf_maxrate0.165_ratio1.0.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset safesurf2025 --ratio 1 --maxrate 0.365 --seed "2015-2026" > fp_multi_flows_safesurf_maxrate0.365_ratio1.0.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset safesurf2025 --ratio 0.001 --maxrate 0.165 --seed "2000-2014" > fp_multi_flows_safesurf_maxrate0.165_ratio1e-3.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset safesurf2025 --ratio 0.001 --maxrate 0.365 --seed "2015-2026" > fp_multi_flows_safesurf_maxrate0.365_ratio1e-3.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset safesurf2025 --ratio 0.00001 --maxrate 0.165 --seed "2000-2014" > fp_multi_flows_safesurf_maxrate0.165_ratio1e-5.log 2>&1 &
nohup python ./fp_multi_flows.py --dataset safesurf2025 --ratio 0.00001 --maxrate 0.365 --seed "2015-2026" > fp_multi_flows_safesurf_maxrate0.365_ratio1e-5.log 2>&1 &

pkill -f fp_multi_flows.py
```
