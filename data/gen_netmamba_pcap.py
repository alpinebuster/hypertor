"""
## Data Preprocessing
- Split individual pcap files
For `x` in [`cic_iot2022`, `cross_platform`, `iscx_tor2016`, `iscx_vpn2016`, `ustc_tfc2016`], download corresponding pcap files from their official websites, and then run the following command to execute `Flow Splitting`
```
python dataset_x.py
```

- Porcess all datasets
    - Sample with lower bound and upper bound
    ```
    python dataset_all.py --sample
    ```
    - Execute `Packet Parsing`, `Packet Cropping & Padding`, and `Packet Concatenating`
    ```
    python dataset_all.py --array
    ```
    - Split train, valid, and test sets for fine-tuning
    ```
    python dataset_all.py --split
    ```
    - Merge all datasets for pre-training
    ```
    python dataset_all.py --merge
    ```

## Dependency
- `editcap`: For changing `.pcapng` into `.pcap` formats, see [editcap](https://www.wireshark.org/docs/man-pages/editcap.html).
- `mergecap`: For merging individual pcap files into an integrated one, see [mergecap](https://www.wireshark.org/docs/man-pages/mergecap.html).
- `splitter`: For efficiently splitting traffic flows from raw pcap files, this is a customized tool, see [ShieldGPT/pcap_tool](https://github.com/wangtz19/ShieldGPT/tree/master/pcap_tool)
"""

import os
from tqdm import tqdm
import subprocess

def split_pcap_files():
    splitter = "./splitter"
    datasets = {
        "cicdarknet": {
            "path": "./ISCX-Tor-NonTor-2017/Tor",
            "flow_packet_num": {
                "BROWSING": 60,
                "CHAT": 100,
                "AUDIO": 100,
                "VIDEO": 200,
                "FILE-TRANSFER": 300,
                "MAIL": 300,
                "P2P": 150,
                "VOIP": 100,
            },
        },

        # "tcub": {
        #     "path": "./darknet-dataset-2020/tor",
        #     "flow_packet_num": {
        #         "BROWSING": 60,
        #         "CHAT": 100,
        #         "AUDIO": 100,
        #         "VIDEO": 200,
        #         "FTP": 300,
        #         "MAIL": 300,
        #         "P2P": 150,
        #         "VOIP": 100,
        #     },
        # },

        # "safesurf": {
        #     "path": "./SafeSurf-Darknet-2025/Tor",
        #     "flow_packet_num": {
        #         "BROWSING": 60,
        #         "CHAT": 100,
        #         "AUDIO": 100,
        #         "VIDEO": 200,
        #         "FTP": 300,
        #         "MAIL": 300,
        #         "VOIP": 100,
        #     },
        # },
    }

    for dataset_name, dataset_cfg in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_path = dataset_cfg["path"]
        flow_packet_num_dict = dataset_cfg["flow_packet_num"]

        for label in flow_packet_num_dict.keys():
            if not os.path.exists(dataset_path):
                print(f"  Warning: {dataset_path} does not exist, skipping.")
                continue

            filenames = [
                os.path.join(dataset_path, f)
                for f in os.listdir(dataset_path) # only consider Tor traffic, ignore NonTor
                if f.lower().endswith(".pcap") and label.lower() in f.lower()
            ]
            print(f"  {label}: {len(filenames)} files found")

            for filename in tqdm(filenames):
                flow_root = os.path.join(
                    os.path.basename(
                        os.path.dirname(os.path.dirname(filename))
                    ),
                    "netmamba_ds",
                    f"flows",
                )
                dir_name = os.path.join(flow_root, label)
                os.makedirs(dir_name, exist_ok=True)

                flow_prefix = os.path.basename(filename)[:-len(".pcap")]
                log_path = os.path.join(flow_root, f"{flow_prefix}.log")
                with open(log_path, "w") as f:
                    subprocess.run(
                        f"{splitter} -i '{filename}' -o '{dir_name}' "
                        f"-p {flow_prefix}- -f five_tuple "
                        f"-l {flow_packet_num_dict[label]}",
                        shell=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )

if __name__ == "__main__":
    split_pcap_files()
