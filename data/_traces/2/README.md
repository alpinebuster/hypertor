# Darknet Datasets

## LowPTor

Run at project root dir

```sh
rsync -avzP ./data/ username@server_ip:/imbalanced_traffic_classification/hypertor/data/
tar -cvf ./data_cache.tar folderA folderB folderC
tar -xvf ./tcub_data_cache.tar -C ./tcub

capinfos ./202602011400.pcap
editcap -r ./202602011400.pcap sample_10pct_202602011400.pcap 1-17700000

sudo -v
nohup sudo ~/.pyenv/versions/3.12.12/bin/python ./data/gen_lowptor.py -n EWTD -d safesurf2025 -r 0.001 > /dev/null 2>&1 &

sudo pkill -INT tcpdump 2>/dev/null || true
sudo pkill -TERM tcpreplay 2>/dev/null || true

ps -ef | grep tcpreplay | grep -v grep
pgrep -a tcpreplay
ps -ef | grep tcpdump | grep -v grep
pgrep -a tcpdump
```

## Darkflow

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
