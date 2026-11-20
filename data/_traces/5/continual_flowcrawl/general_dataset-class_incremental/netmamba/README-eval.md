## NonVPN
```sh
CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval.py \
    --nb_classes 2 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/raw/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/raw/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-raw.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval.py \
    --nb_classes 3 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/raw_ss/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-raw_ss.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval.py \
    --nb_classes 4 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/raw_ss_trojan/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss_trojan/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss_trojan/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-raw_ss_trojan.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval.py \
    --nb_classes 5 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-raw_ss_trojan_vless.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval.py \
    --nb_classes 6 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-raw_ss_trojan_vless_vmess.log 2>&1 &
```

## VPN
```sh
CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval.py \
    --nb_classes 2 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/VPN_raw/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-VPN_raw.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval.py \
    --nb_classes 3 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/VPN_raw_ss/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-VPN_raw_ss.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval.py \
    --nb_classes 4 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss_trojan \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-VPN_raw_ss_trojan.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval.py \
    --nb_classes 5 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-VPN_raw_ss_trojan_vless.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval.py \
    --nb_classes 6 \
    --batch_size 32 \
    --resume ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess/output-eval \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess/logs-eval \
    --model net_mamba_classifier \
> dataset_sampled-class_incremental-VPN_raw_ss_trojan_vless_vmess.log 2>&1 &
```
