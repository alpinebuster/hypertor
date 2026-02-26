# DARKFLOW DATASETS

## Getting Started

At project's root dir:

```sh
rsync -avzP ./data/  username@server_ip:/imbalanced_traffic_classification/hypertor/data/
tar -cvf ./tcub_data_cache.tar ./data/tcub/cache/
find . -type f -name "*merged_5d.npy"
sudo mount -t exfat -o rw,uid=1000,gid=1000 /dev/sdc2 /mnt/usb
sudo umount /mnt/usb
```
