# CONTINUAL DARKFLOW DATASETS

The **CONTINUAL DARKFLOW DATASETS** are designed for continual learning research on encrypted traffic analysis and darknet traffic understanding. The datasets support both:

* **Binary detection tasks** for distinguishing normal and anonymized traffic.
* **Multi-class classification tasks** for identifying different proxy and tunneling protocols.

The dataset is constructed from flow-level traffic sequences and uses normalized temporal and packet-level statistical features for model training and evaluation.

---

## 1. Detection Task

Binary detection labels:

| Index | Label        |
| ----: | ------------ |
|     0 | nontor       |
|     1 | tor-over-vpn |

---

## 2. Classification Task

Traffic class mapping:

| Class ID | Label      |
| -------: | ---------- |
|        0 | ss_pos     |
|        1 | tor_pos    |
|        2 | trojan_pos |
|        3 | vless_pos  |
|        4 | vmess_pos  |

---

## 3. Features Used

The following packet-sequence features are used:

```python
[
    "dir_seq",
    "norm_len",
    "norm_usec",
    "norm_iat",
    "signed_norm_len",
    "signed_norm_usec",
]
```

### Feature Description

| Feature Name       | Description                                |
| ------------------ | ------------------------------------------ |
| `dir_seq`          | Packet direction sequence                  |
| `norm_len`         | Normalized packet length                   |
| `norm_usec`        | Normalized packet timestamp (microseconds) |
| `norm_iat`         | Normalized inter-arrival time              |
| `signed_norm_len`  | Signed normalized packet length            |
| `signed_norm_usec` | Signed normalized timestamp                |

## Getting Started

At project's root dir:

```sh
rsync -avzP ./data/ username@server_ip:/imbalanced_traffic_classification/hypertor/data/
tar -cvf ./tcub_data_cache.tar ./data/tcub/cache/
tar -xvf ./tcub_data_cache.tar -C ./tcub
find . -type f -name "*merged_5d.npy"

# (Optional)
parted /dev/sdb
mklabel gpt
mkpart primary 0% 100%
quit
partprobe

lsblk
>>> /dev/sdb1
sudo mkfs.ext4 /dev/sdb1
sudo mount /dev/sdb1 /mnt/data

sudo vim /etc/fstab
>>> Add `/dev/sdb1   /mnt/data   ext4   defaults   0  0`
# or
lsblk -f
>>> Add `UUID=db6ccb17-23b2-4506-82fe-8afb5543aff0  /mnt/data  ext4  defaults  0  0`

# Test auto mount after boot
mount -a

sudo mount -t exfat -o rw,uid=1000,gid=1000 /dev/sdc2 /mnt/usb
sudo umount /mnt/usb
```
