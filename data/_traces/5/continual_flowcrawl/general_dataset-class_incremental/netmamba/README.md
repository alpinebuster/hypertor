## NonVPN
```sh
CUDA_VISIBLE_DEVICES=0 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 2 \
    --batch_size 32 \
    --finetune ./pre-train.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw/output \
    --log_dir dataset/dataset_sampled-class_incremental/raw/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-raw.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 3 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/raw/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss/output \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-raw_ss.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 4 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/raw_ss/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss_trojan \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss_trojan/output \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss_trojan/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-raw_ss_trojan.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 5 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/raw_ss_trojan/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless/output \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-raw_ss_trojan_vless.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 6 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess \
    --output_dir ./dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess/output \
    --log_dir dataset/dataset_sampled-class_incremental/raw_ss_trojan_vless_vmess/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-raw_ss_trojan_vless_vmess.log 2>&1 &
```

## VPN
```sh
CUDA_VISIBLE_DEVICES=1 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 2 \
    --batch_size 32 \
    --finetune ./pre-train.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw/output \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-VPN_raw.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 3 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/VPN_raw/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss/output \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-VPN_raw_ss.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 4 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/VPN_raw_ss/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan/output \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-VPN_raw_ss_trojan.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 5 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless/output \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-VPN_raw_ss_trojan_vless.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes 6 \
    --batch_size 32 \
    --finetune ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless/output/checkpoint-best.pth \
    --data_path ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess \
    --output_dir ./dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess/output \
    --log_dir dataset/dataset_sampled-class_incremental/VPN_raw_ss_trojan_vless_vmess/logs \
    --model net_mamba_classifier \
    --no_amp \
> dataset_sampled-class_incremental-VPN_raw_ss_trojan_vless_vmess.log 2>&1 &
```
