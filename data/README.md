# DARKFLOW DATASETS

## Getting Started

At project's root dir:

```sh
rsync -avzP ./data/ username@server_ip:/imbalanced_traffic_classification/hypertor/data/
tar -cvf ./tcub_data_cache.tar ./data/tcub/cache/
tar -xvf ./tcub_data_cache.tar -C ./tcub

find . -type f -name "*merged_5d.npy"
ls -l | grep ^- | wc -l
find /path/to/folder -type f | wc -l

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

### Darkflow

```sh
sudo ../darkflow/target/release/darkflow

# Change to project root dir
cd ..
# REF: `../libs/cicflowmeter/README.md`
git clone https://github.com/hieulw/cicflowmeter
cd cicflowmeter
uv sync
source .venv/bin/activate

nohup cicflowmeter -d ./data/LOWPTOR/cicdarknet/voip/ -c ./data/LOWPTOR/cicdarknet/ > cicdarknet-voip.log 2>&1 &
nohup cicflowmeter -d ./data/LOWPTOR/safesurf2025/voip/ -c ./data/LOWPTOR/safesurf2025/ > safesurf2025-voip.log 2>&1 &
```
