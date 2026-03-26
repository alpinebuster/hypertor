import os
from tqdm import tqdm
import subprocess
import json
from PIL import Image
import random
import numpy as np
import shutil
import argparse
import binascii
import scapy.all as scapy


def find_files(data_path, extension=".pcap"):
    pcap_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def raw_packet_to_string(packet, remove_ip=True, keep_payload=True):
    ip = packet["IP"]
    if remove_ip:
        PAD_IP_ADDR = "0.0.0.0"
        ip.src, ip.dst = PAD_IP_ADDR, PAD_IP_ADDR
    header = (binascii.hexlify(bytes(ip))).decode()
    if keep_payload:
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
    else:
        payload = ''
    header = header[:160] if len(header) > 160 else header + '0' * (160 - len(header))
    payload = payload[:480] if len(payload) > 480 else payload + '0' * (480 - len(payload))
    return header, payload

def string_to_hex_array(flow_string):
    return np.array([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)])

def read_5hp_list(pcap_filename, if_augment=False, remove_ip=True, keep_payload=True):
    packets = scapy.rdpcap(pcap_filename)
    data = []
    flow_string_length = 3200
    flow_packet_num = 5
    end = len(packets) if if_augment else flow_packet_num
    for packet in packets[:end]:
        try:
            # ip = packet['IP']
            header, payload = raw_packet_to_string(packet, remove_ip=remove_ip, keep_payload=keep_payload)
        except:
            # continue
            header, payload = '0' * 160, '0' * 480
        data.append(header + payload)

    if not if_augment or len(data) <= flow_packet_num:
        flow_string = ''.join(data)
        flow_string += '0' * (flow_string_length - len(flow_string))
        flow_array = string_to_hex_array(flow_string)
        return [{
            "data": flow_array,
        }]
    else:
        assert len(data) > flow_packet_num
        flow_array_list = []
        for i in range(len(data) - flow_packet_num + 1):
            flow_string = ''.join(data[i:i + flow_packet_num])
            flow_array_list.append(string_to_hex_array(flow_string))
        return [{
            "data": flow_array,
        } for flow_array in 
        zip(flow_array_list)]


def _sample_pcap(
    minimum=200,
    maximum=3000,
    input_dir="./ISCX-Tor-NonTor-2017/netmamba_ds/flows",
    output_dir="./ISCX-Tor-NonTor-2017/netmamba_ds/flows_sampled",
    if_cic=False,
):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    random.seed(0)
    os.makedirs(output_dir, exist_ok=True)
    sub_dirs = list(filter(lambda x: os.path.isdir(f"{input_dir}/{x}"), os.listdir(input_dir)))
    if if_cic:
        pcap_files = []
        for sub_dir in sub_dirs:
            pcap_files.extend(find_files(f"{input_dir}/{sub_dir}", extension=".pcap"))
        if len(pcap_files) < minimum: # Skip if less than minimum
            print(f"Skip {input_dir} due to less than {minimum} flows")
            return
        if len(pcap_files) > maximum:
            pcap_files = random.sample(pcap_files, maximum)
        for pcap_file in tqdm(pcap_files, desc=output_dir):
            compressed_name = pcap_file[len(input_dir):].replace("/", "_")
            dst_pcap_file = f"{output_dir}/{compressed_name}"
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)
    else:
        for sub_dir in sub_dirs:
            pcap_files = find_files(f"{input_dir}/{sub_dir}", extension=".pcap")
            if len(pcap_files) < minimum: # Skip if less than minimum
                print(f"Skip {input_dir}/{sub_dir} due to less than {minimum} flows")
                continue
            if len(pcap_files) > maximum:
                pcap_files = random.sample(pcap_files, maximum)
            for pcap_file in tqdm(pcap_files, desc=f"{output_dir}/{sub_dir}"):
                dst_pcap_file = pcap_file.replace(input_dir, output_dir)
                os.makedirs("/".join(dst_pcap_file.split("/")[:-1]), exist_ok=True)
                subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)

def sample_all_pcap():
    # _sample_pcap(
    #     input_dir="./darknet-dataset-2020/netmamba_ds/flows", 
    #     output_dir=f"./darknet-dataset-2020/netmamba_ds/flows_sampled",
    #     minimum=10, maximum=4000,
    # )
    _sample_pcap(
        input_dir="./ISCX-Tor-NonTor-2017/netmamba_ds/flows",
        output_dir=f"./ISCX-Tor-NonTor-2017/netmamba_ds/flows_sampled",
        minimum=10, maximum=4000,
    )
    # _sample_pcap(
    #     input_dir="./SafeSurf-Darknet-2025/netmamba_ds/flows",
    #     output_dir=f"./SafeSurf-Darknet-2025/netmamba_ds/flows_sampled",
    #     minimum=10, maximum=4000,
    # )


def _pcap_to_array(pcap_dir, if_augment=False):
    assert pcap_dir.split("/")[-1] == "flows_sampled"
    image_dir = pcap_dir.replace("flows_sampled", "array_sampled")
    flow_dir_names = os.listdir(pcap_dir)
    for flow_dir_name in flow_dir_names:
        os.makedirs(f"{image_dir}/{flow_dir_name}", exist_ok=True)
        pcap_filenames = os.listdir(f"{pcap_dir}/{flow_dir_name}")
        for pcap_filename in tqdm(pcap_filenames, desc=flow_dir_name):
            try:
                if not if_augment:
                    image_filename = f"{image_dir}/{flow_dir_name}/{pcap_filename[:-len('.pcap')]}.png"
                    stat_filename = image_filename.replace(".png", ".json")
                    res = read_5hp_list(f"{pcap_dir}/{flow_dir_name}/{pcap_filename}")[0]
                    flow_array = res.pop("data")
                    image = Image.fromarray(flow_array.reshape(40, 40).astype(np.uint8))
                    image.save(image_filename)
                    with open(stat_filename, "w") as f:
                        json.dump(res, f)
                else:
                    res_list = read_5hp_list(f"{pcap_dir}/{flow_dir_name}/{pcap_filename}", if_augment=True)
                    for i, res in enumerate(res_list):
                        image_filename = f"{image_dir}/{flow_dir_name}/{pcap_filename[:-len('.pcap')]}-{i}.png"
                        stat_filename = image_filename.replace(".png", ".json")
                        flow_array = res.pop("data")
                        image = Image.fromarray(flow_array.reshape(40, 40).astype(np.uint8))
                        image.save(image_filename)
                        with open(stat_filename, "w") as f:
                            json.dump(res, f)
            except Exception as e:
                print(f"Error processing {pcap_filename}: {e}")

def all_pcap_to_array():
    # _pcap_to_array("./darknet-dataset-2020/netmamba_ds/flows_sampled", if_augment=True)
    _pcap_to_array("./ISCX-Tor-NonTor-2017/netmamba_ds/flows_sampled", if_augment=True)
    # _pcap_to_array("./SafeSurf-Darknet-2025/netmamba_ds/flows_sampled", if_augment=True)


def _split_dataset(input_dir, train_ratio=0.8, valid_ratio=0.1):
    assert input_dir.split("/")[-1] == "array_sampled"
    dir_name = "dataset_sampled"
    output_dir = input_dir.replace("array_sampled", dir_name)
    train_dir = f"{output_dir}/train"
    valid_dir = f"{output_dir}/valid"
    test_dir = f"{output_dir}/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(0)
    filenames = find_files(input_dir, extension=".png")
    np.random.shuffle(filenames)
    train_size = int(len(filenames) * train_ratio)
    valid_size = int(len(filenames) * valid_ratio)
    train_files = filenames[:train_size]
    valid_files = filenames[train_size:train_size+valid_size]
    test_files = filenames[train_size+valid_size:]
    for filename in tqdm(filenames, desc="Splitting"):
        if filename in train_files:
            split_type = "train"
        elif filename in valid_files:
            split_type = "valid"
        else:
            split_type = "test"
        label, base_name = filename.split("/")[-2:]
        os.makedirs(f"{output_dir}/{split_type}/{label}", exist_ok=True)
        subprocess.run(f"cp '{filename}' '{output_dir}/{split_type}/{label}/{base_name}'", shell=True)
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

def split_all_datasets():
    datasets = [
        # "darknet-dataset-2020",
        "ISCX-Tor-NonTor-2017",
        # "SafeSurf-Darknet-2025",
    ]
    for dataset in datasets:
        _split_dataset(f"{dataset}/netmamba_ds/array_sampled")


def merge_dataset():
    filenames = []
    datasets = [
        "darknet-dataset-2020",
        "ISCX-Tor-NonTor-2017",
        "SafeSurf-Darknet-2025",
    ]
    for dataset in datasets:
        filenames += find_files(f"{dataset}/array_sampled", extension=".png")
    for filename in tqdm(filenames, desc="Merging"):
        filename_list = filename.split("/")
        label = filename_list[0] + "-" + filename_list[-2]
        base_name = filename_list[-1]
        os.makedirs(f"pretrain_dataset/train/{label}", exist_ok=True)
        dst_filename = f"pretrain_dataset/train/{label}/{base_name}"
        subprocess.run(f"cp '{filename}' '{dst_filename}'", shell=True)


def _get_lowp_dataset(input_dir: str, datasetname: str):
    DATASET_RATIO_SAMPLES = {
        "cicdarknet": {
            0.001: [57, 5, 28, 10, 9, 15, 6],
            1e-05: [4, 5, 7, 5, 5, 5, 5],
        },
        "tcub": {
            0.001: [13, 17, 14, 45, 68, 9, 52],
            1e-05: [4, 4, 4, 2, 9, 6, 2],
        },
        "safesurf2025": {
            0.001: [194, 159, 162, 127, 189, 238],
            1e-05: [12, 9, 18, 15, 13, 43],
        },
    }

    splits = ["train", "test", "valid"]
    parent_dir = os.path.dirname(input_dir)
    for ratio, samples in DATASET_RATIO_SAMPLES[datasetname].items():
        out_dir = os.path.join(parent_dir, f"dataset_sampled_ratio{ratio}")
        os.makedirs(out_dir, exist_ok=True)

        for split in splits:
            split_dir = os.path.join(input_dir, split)
            out_split_dir = os.path.join(out_dir, split)
            os.makedirs(out_split_dir, exist_ok=True)

            classes = sorted(os.listdir(split_dir))
            for cls_idx, clsname in enumerate(classes):
                src_cls_dir = os.path.join(split_dir, clsname)
                dst_cls_dir = os.path.join(out_split_dir, clsname)
                os.makedirs(dst_cls_dir, exist_ok=True)

                files = os.listdir(src_cls_dir)
                n = samples[cls_idx]
                if split != "train":
                    n = (n + 1) // 2
                n = min(n, len(files))

                selected = random.sample(files, n)
                for f in selected:
                    src = os.path.join(src_cls_dir, f)
                    dst = os.path.join(dst_cls_dir, f)
                    shutil.copy2(src, dst)

        print(f"Finished sampling ratio {ratio} -> {out_dir}")

def get_all_lowp():
    datasets = {
        # "tcub": "darknet-dataset-2020",
        "cicdarknet": "ISCX-Tor-NonTor-2017",
        # "safesurf2025": "SafeSurf-Darknet-2025",
    }
    for dataset_key, dataset_value in datasets.items():
        _get_lowp_dataset(
            f"{dataset_value}/netmamba_ds/dataset_sampled",
            dataset_key,
        )


if __name__ == "__main__":
    """
    e.g. `nohup python ./gen_netmamba_ds.py --sample --array --split > ./gen_netmamba_ds.log 2>&1 &`
         `nohup python ./gen_netmamba_ds.py --lowp > ./gen_netmamba_ds-lowp.log 2>&1 &`
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Sample pcap files")
    parser.add_argument("--array", action="store_true", help="Convert pcap files to array")
    parser.add_argument("--split", action="store_true", help="Split dataset into train, valid, and test sets for finetuning")
    parser.add_argument("--lowp", action="store_true", help="Get lowp dataset")
    parser.add_argument("--merge", action="store_true", help="Merge all datasets into one for pretraining")

    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    RATIO = args.ratio
    # import pdb;pdb.set_trace()
    if args.sample or args.all:
        sample_all_pcap()
    if args.array or args.all:
        all_pcap_to_array()
    if args.split or args.all:
        split_all_datasets()
    if args.lowp or args.all:
        get_all_lowp()
    if args.merge or args.all:
        merge_dataset()
