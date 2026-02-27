# DARKFLOW DATASETS

## Getting Started

At project's root dir:

```sh
rsync -avzP ./data/  username@server_ip:/imbalanced_traffic_classification/hypertor/data/
tar -cvf ./tcub_data_cache.tar ./data/tcub/cache/
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
