# Imbalanced Traffic Classification

In this study, we propose a comprehensive framework that integrates multiple components for traffic classification.

**1. Multi-Flow Fingerprinting Mechanism**
To enhance feature representation and effectively distinguish between traffic classes, we propose a bespoke fingerprinting mechanism. This method aggregates multi flows originating from a single access event, enabling richer and more discriminative traffic representations.

**2. TCP-Sequence-Driven Data Augmentation**
To alleviate the limitations imposed by scarce training data, we introduce a TCP-sequence-driven data augmentation strategy. This approach leverages intrinsic TCP sequence characteristics to generate meaningful augmented samples while preserving semantic consistency.

**3. Variable-Prototype Learning for Class Imbalance**
To handle class imbalance, we propose a variable-prototype learning paradigm that:

- Adaptively determines the number of prototypes for each class  
- Simultaneously learns data-driven class centers  

This design produces more discriminative decision boundaries and improves classification robustness, particularly for minority classes.

**4. Transformer-Based Layerwise Hypernetwork**
We employ a Transformer-based layerwise hypernetwork, in which:

- A Transformer architecture generates the parameters of Convolutional Neural Networks (CNNs) & Variable Prototypes
- Only hypernetwork weights are shared across participants  

This strategy enhances model flexibility while preserving data privacy.

## Getting started

### Python Poetry
```sh
poetry install

poetry env activate
# or
source $(poetry env info -p)/bin/activate
# or
. $(poetry env info -p)/bin/activate

poetry env remove --all
```

### Tor
```sh
sudo apt install tor obfs4proxy

# UseBridges 1
# ClientTransportPlugin obfs4 exec /usr/bin/obfs4proxy managed
# Bridge obfs4 57.128.68.206:12069 3D31D10FFF77231A352CA982BD373ED9235FD769 cert=xPPp3sNp0xVeoxF+SL7ixf7cefyotL7eBCgPctKyqosvDZMKD2TvvEccUcR5IFkJyM4qFQ iat-mode=0
# Bridge obfs4 91.134.80.41:59987 ACBB6AD034A3C0712265F1BFEC9F27D33B7EFC20 cert=GjxluYzjUz3H1aPRnwncHk46UYhFXvWjn4SA1wwFTfIAFVuBsfsvuZA5KlUi4obxcSaeIQ iat-mode=0
sudo vi /etc/tor/torrc

sudo systemctl enable tor
sudo systemctl restart tor
sudo journalctl -e -f -u tor@default
```

### Running Examples

Folder `scripts` contains several training scripts for `darkflow`,
`darknet-dataset-2020` and `ISCX-Tor-NonTor-2017` datasets.

They can be executed as follows:

```shell
./scripts/darkflow_1shot.sh --data_dir=./data/darkflow --data_numpy_dir=./data/darkflow/cache --train_log_dir=./data/darkflow/logs/torch
```

The evaluation script `evaluate_model.py` can be executed in parallel to
evaluate produced checkpoints while the model is being trained.
The evaluation script accepts the same parameters as the training script.

> DDP (DistributedDataParallel) Supports

```sh
# (Optional)
sudo sysctl -w vm.max_map_count=262144
>>> vm.max_map_count = 262144
cat /proc/sys/vm/max_map_count
>>> 262144

sudo sysctl -w vm.overcommit_memory=1
>>> vm.overcommit_memory = 1
sudo sysctl -w vm.overcommit_ratio=100
>>> vm.overcommit_ratio = 100

cat /proc/sys/vm/overcommit_memory
>>> 1
cat /proc/sys/vm/overcommit_ratio
>>> 100

swapon --show
>>> NAME      TYPE SIZE USED PRIO
>>> /swapfile file   2G   0B   -2
sudo fallocate -l 32G /swapfile32
sudo chmod 600 /swapfile32
sudo mkswap /swapfile32
>>> Setting up swapspace version 1, size = 32 GiB (34359734272 bytes)
>>> no label, UUID=42979462-b31e-4ef1-8d2f-b0772f9e95ad
sudo swapon /swapfile32
swapon --show
free -h
sudo nano /etc/fstab # Add "/swapfile32 none swap sw 0 0" to it
cat /proc/sys/vm/swappiness
sudo sysctl -w vm.swappiness=10
sudo nano /etc/sysctl.conf # Add "vm.swappiness=10" to it

sudo slabtop

# `nproc_per_node` -> Number of GPUs
# `--` -> All the parameters that follow are not `torchrun` but passed to the python
nohup torchrun --nproc_per_node=2 ./hypertor/train.py -- \
  --num_layerwise_features=8 --default_num_channels=8 \
  --samples_transformer=20 --samples_cnn=60 --num_labels=20 --learning_rate=0.02 \
  --learning_rate_decay_steps=100000.0 --learning_rate_decay_rate=0.95 \
  --train_steps=4000000 --steps_between_saves=1000 --lw_key_query_dim=1.0 \
  --lw_value_dim=1.0 --lw_inner_dim=1.0 --cnn_model_name='maxpool-4-layer' \
  --embedding_dim=32 --num_layers=3 --stride=1 --heads=2 \
  --shared_feature_extractor='3-layer-bn' --shared_input_dim=1 --shared_features_dim=32 \
  --shared_feature_extractor_padding=same --layerwise_generator=joint \
  --nolw_use_nonlinear_feature --lw_weight_allocation=output --nolw_generate_bias \
  --nolw_generate_bn --nouse_decoder --noadd_trainable_weights --flows_shape=28,28 \
  --balanced_batches --per_label_augmentation \
  --smooth_probability=0.0 --negate_probability=0.0 \
  --train_dataset='omniglot:0-1149' \
  --test_dataset='omniglot:1200-1622' --eval_datasets='omniglot' \
  --num_task_evals=1024 --num_eval_batches=16 --eval_batch_size=100 \
  --shuffle_labels_seed=2026 \
  --test_smooth_probability=0.0 --test_negate_probability=0.0 \
  --data_dir=./data/omniglot --data_numpy_dir=./data/omniglot/cache \
  --train_log_dir=./data/omniglot/logs/torch_ddp \
  --use_ddp \
  > omniglot_ddp.log 2>&1 &

echo $?
```

### HyperTor

The code consists of several parts:

1. _Task Generator_ — uses input dataset to generate _episodes_.
2. _Transformer IO (Input)_ — receives a support batch (typically a few samples
   per label) and encodes it producing sample embeddings that are passed to the
   Transformer model.
3. _Transformer_ — receives a set of input embeddings (encoding input samples)
   and produces a set of output embeddings (containing CNN model weights).
4. _Transformer IO (Output)_ — converts Transformer output to a list of tensors
   that contain unstructured CNN model weights.
5. _CNN Model Builder_ — uses generated weight tensors to create a CNN.

The model is trained end-to-end:
```txt
   ┌───────────────────────────┐
   │ (a) Generate a batch of   │
   │     samples for a new     │<┐
   │         episode           │ │
   └─────────────┬─────────────┘ │
                 │               │
                 ▼               │
   ┌───────────────────────────┐ │
   │ (b) Pass support samples  │ │
   │     to Transformer →      │ │
   │     Transformer generates │ │
   │           CNN             │ │
   └─────────────┬─────────────┘ │
                 │               │
                 ▼               │
   ┌───────────────────────────┐ │
   │ (c) Pass query samples    │ │
   │     through generated CNN │ │ Update Prototypes
   │     and compute loss →    │─┘
   │     Backprop to train     │
   │       Transformer         │
   └───────────────────────────┘
                 │
                 ▼
```
