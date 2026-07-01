import os
import attr
import csv
import glob
from typing import Tuple
from pathlib import Path
import logging
import random
import pickle
from typing import TYPE_CHECKING, List, Tuple
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP

from hypertor.core.helpers import get_inter_diff_by_stream, PacketDirection, TCPFlagNum, \
    replenish_list


class _TCPFlags:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _TCPLayer:
    __slots__ = ("flags",)

    def __init__(self, flags):
        self.flags = _TCPFlags(flags)


@dataclass(slots=True)
class PacketMeta:
    time: float
    direction: int
    length: int
    tcp_flags: int

    def __len__(self):
        return self.length

    def __getitem__(self, layer):
        if layer is TCP:
            return _TCPLayer(self.tcp_flags)
        raise KeyError(layer)


class Pcap2Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.UNWANTED_CLASSES = self.config.UNWANTED_CLASSES
        self.samples = []
        self.labels = []
        cache_path = os.path.join(
            config.DATASET_PATH,
            "pcap2dataset_cache.pkl",
        )

        if os.path.exists(cache_path):
            logging.info("Loading by using .pkl cache")

            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            self.samples = data["samples"]
            self.labels = data["labels"]
            self.class_to_idx = data["class_to_idx"]
            self.idx_to_class = data["idx_to_class"]
            self.per_class_counts = data["per_class_counts"]

            self.num_classes = len(self.class_to_idx)
        else:
            logging.info("Loading by handling .pcap files")
            class_names = sorted([
                d for d in os.listdir(config.DATASET_PATH)
                if os.path.isdir(os.path.join(config.DATASET_PATH, d))
                and "_biflows" not in d
                and "_log" not in d
                and "_uniflows" not in d
                and d not in self.UNWANTED_CLASSES
            ])
            self.class_to_idx = {
                class_name: idx
                for idx, class_name in enumerate(class_names)
            }
            self.idx_to_class = {
                idx: class_name
                for class_name, idx in self.class_to_idx.items()
            }
            self.num_classes = len(class_names)
            logging.info(f"class_to_idx: {self.class_to_idx}")

            for class_name, label in self.class_to_idx.items():
                folder_path = os.path.join(config.DATASET_PATH, class_name)

                for root, _, files in os.walk(folder_path):
                    for f in files:
                        if not f.endswith(".pcap"):
                            continue

                        pcap_path = os.path.join(root, f)
                        logging.info(f"pcap_path: {pcap_path}")
                        _, flows_raw = handle_pcap(str(pcap_path))
                        for _, raw_flow in flows_raw.items():
                            if len(raw_flow) < 30:
                                continue

                            self.samples.append((raw_flow, label))
                            self.labels.append(label)

            self.per_class_counts = np.bincount(
                self.labels,
                minlength=self.num_classes,
            )
            logging.info(f"self.per_class_counts: {self.per_class_counts}")

            # Delete the categories with too few samples
            MIN_SAMPLES_PER_CLASS = 20
            valid_labels = {
                label
                for label, cnt in enumerate(self.per_class_counts)
                if cnt >= MIN_SAMPLES_PER_CLASS
            }
            label_mapping = {
                old_label: new_label
                for new_label, old_label in enumerate(sorted(valid_labels))
            }

            new_samples = []
            new_labels = []
            for raw_flow, old_label in self.samples:
                if old_label not in valid_labels:
                    continue

                new_label = label_mapping[old_label]
                new_samples.append((raw_flow, new_label))
                new_labels.append(new_label)

            self.samples = new_samples
            self.labels = new_labels

            self.class_to_idx = {
                class_name: label_mapping[label]
                for class_name, label in self.class_to_idx.items()
                if label in valid_labels
            }
            self.idx_to_class = {
                v: k
                for k, v in self.class_to_idx.items()
            }
            self.num_classes = len(self.class_to_idx)
            logging.info(f"self.class_to_idx: {self.class_to_idx}")

            self.per_class_counts = np.bincount(
                self.labels,
                minlength=self.num_classes,
            )

            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "samples": self.samples,
                        "labels": self.labels,
                        "class_to_idx": self.class_to_idx,
                        "idx_to_class": self.idx_to_class,
                        "per_class_counts": self.per_class_counts,
                    },
                    f,
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_flow, label = self.samples[idx]
        features = self.load_flow(raw_flow)
        return features, torch.tensor(label, dtype=torch.long)

    def load_flow(self, raw_flow):
        iats_up, iats_down, iats_global = get_sign_inter_iat(raw_flow, self.config.INPUT_FLOW_LEN)
        (
            inter_payload_diff_up,
            inter_payload_diff_down,
            inter_payload_diff_global
        ) = get_sign_inter_payload_diff(raw_flow, self.config.INPUT_FLOW_LEN)

        SIGN = replenish_list(get_sign(raw_flow, self.config.INPUT_FLOW_LEN))
        SIGN_PKT_LEN = replenish_list(get_sign_pktlength(raw_flow, self.config.INPUT_FLOW_LEN))
        SIGN_FLAG = replenish_list(get_sign_flag(raw_flow, self.config.INPUT_FLOW_LEN))

        IAT_UP = replenish_list(iats_up)
        IAT_DOWN = replenish_list(iats_down)
        IAT_GLOBAL = replenish_list(iats_global)

        INTER_PAYLOAD_DIFF_UP = replenish_list(inter_payload_diff_up)
        INTER_PAYLOAD_DIFF_DOWN = replenish_list(inter_payload_diff_down)
        INTER_PAYLOAD_DIFF_GLOBAL = replenish_list(inter_payload_diff_global)

        flow_fe = np.array([
            SIGN,
            SIGN_PKT_LEN,
            SIGN_FLAG,
            IAT_UP,
            IAT_DOWN,
            IAT_GLOBAL,
            INTER_PAYLOAD_DIFF_UP,
            INTER_PAYLOAD_DIFF_DOWN,
            INTER_PAYLOAD_DIFF_GLOBAL,
        ], dtype=np.float32)

        flow_fe = np.nan_to_num(flow_fe, nan=0.0, posinf=0.0, neginf=0.0)
        flow_fe = flow_fe[:, :self.config.INPUT_FLOW_LEN]

        # ===== split =====
        sign = flow_fe[0]
        SIGN_UP = np.where(sign >= 0, sign, 0)
        SIGN_DOWN = np.abs(np.where(sign < 0, sign, 0))

        sign_pkt_len = flow_fe[1]
        SIGN_PKT_LEN_UP = np.where(sign_pkt_len >= 0, sign_pkt_len, 0)
        SIGN_PKT_LEN_DOWN = np.abs(np.where(sign_pkt_len < 0, sign_pkt_len, 0))

        sign_flag = flow_fe[2]
        SIGN_FLAG_UP = np.where(sign_flag >= 0, sign_flag, 0)
        SIGN_FLAG_DOWN = np.abs(np.where(sign_flag < 0, sign_flag, 0))

        up_channels = np.stack([SIGN_UP, SIGN_PKT_LEN_UP, SIGN_FLAG_UP], axis=0)
        down_channels = np.stack([SIGN_DOWN, SIGN_PKT_LEN_DOWN, SIGN_FLAG_DOWN], axis=0)
        global_channels = np.stack([sign, np.abs(sign_pkt_len), np.abs(sign_flag)], axis=0)

        fe_data = np.stack([up_channels, down_channels, global_channels], axis=0)

        fe_data = np.ascontiguousarray(fe_data)
        return torch.tensor(fe_data.copy(), dtype=torch.float32)


def get_flow_key(pkt):
    if IP not in pkt:
        return None

    ip = pkt[IP]

    proto = None
    sport = dport = 0

    if TCP in pkt:
        proto = "TCP"
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif UDP in pkt:
        proto = "UDP"
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        proto = ip.proto

    return (ip.src, sport, ip.dst, dport, proto)


def normalize_direction(flow_key, pkt_key):
    if flow_key[:2] == pkt_key[:2]:
        return 1
    return -1


def handle_pcap(pcap_path: str):
    packets = rdpcap(pcap_path)

    flows = defaultdict(list)
    flows_raw = defaultdict(list)
    for pkt in packets:
        if IP not in pkt:
            continue
        if TCP not in pkt:
            continue

        flow_key = get_flow_key(pkt)
        if flow_key is None:
            continue

        ts = float(pkt.time)
        length = len(pkt)
        # import pdb;pdb.set_trace()

        src, sport, dst, dport, proto = flow_key
        forward_key = (src, sport, dst, dport, proto)
        reverse_key = (dst, dport, src, sport, proto)

        if forward_key in flows:
            direction = 1
            key = forward_key
        elif reverse_key in flows:
            direction = -1
            key = reverse_key
        else:
            direction = 1
            key = forward_key

        flows[key].append((ts, direction, length))
        if direction == 1:
            pkt.direction = PacketDirection.UPSTREAM # type:ignore
        else:
            pkt.direction = PacketDirection.DOWNSTREAM # type:ignore
        # flows_raw[key].append(pkt)
        flows_raw[key].append(
            PacketMeta(
                time=ts,
                direction=direction,
                length=length,
                tcp_flags=pkt[TCP].flags.value,
            )
        )

    # Sort by timestamp
    # for k in flows:
    #     flows[k].sort(key=lambda x: x[0])

    for k, v in list(flows.items())[:3]:
        print("  FLOW:", k)
        # for item in v:
        #     print("   ts:", item[0], "dir:", item[1], "len:", item[2])

    return flows, flows_raw


def get_sign(curr_flow, pkt_num_threshold: int = 30) -> list:
    sign_pkt_list = []

    for idx, pkt in enumerate(curr_flow):
        if idx < pkt_num_threshold:
            sign_pkt_list.append(
                1
                if pkt.direction == PacketDirection.UPSTREAM
                else -1
            )
        else:
            break

    return sign_pkt_list


def get_sign_pktlength(curr_flow, pkt_num: int = 30) -> list:
    """Collects the lengths of packets from the current flow,
    indicating their direction with positive or negative values.
    It limits the number of packets processed to the specified pkt_num,
    lexnet is set to 20, and tolnet is set to 30 by default.

    Args:
        pkt_num (int, optional): Only the first pkt_num packets will be processed.

    Returns:
        list: The resulting list can be useful for analyzing the flow of packets in terms of
            their sizes and directions.
    """
    sign_pkt_length_list = []

    for idx, pkt in enumerate(curr_flow):
        if idx < pkt_num:
            sign_pkt_length_list.append(
                len(pkt)
                if pkt.direction == PacketDirection.UPSTREAM
                else -len(pkt)
            )
        else:
            break

    return sign_pkt_length_list


def get_sign_inter_iat(
    curr_flow, pkt_num_threshold: int = 30
) -> Tuple[List[float], List[float], List[float]]:
    pkts_up: List[float] = []
    pkts_down: List[float] = []
    pkts_global: List[float] = []

    # The first value, 0, is used for padding to the length of the flow
    iats_up: List[float] = []
    iats_down: List[float] = []
    iats_global: List[float] = [0.]

    for idx, pkt in enumerate(curr_flow):
        if idx < pkt_num_threshold:
            # import pdb;pdb.set_trace()
            pkts_global.append(pkt.time)

            # print(f"pkt.direction: {pkt.direction}")
            if pkt.direction == PacketDirection.UPSTREAM:
                pkts_up.append(pkt.time)
                pkts_down.append(0.)
            else:
                pkts_down.append(pkt.time)
                pkts_up.append(0.)

    iats_up = get_inter_diff_by_stream(pkts_up)
    iats_down = get_inter_diff_by_stream(pkts_down)
    iats_global = get_inter_diff_by_stream(pkts_global)

    return iats_up, iats_down, iats_global


def get_sign_inter_payload_diff(
    curr_flow, pkt_num_threshold: int = 30
) -> Tuple[List[float], List[float], List[float]]:
    pkts_up: List[float] = []
    pkts_down: List[float] = []
    pkts_global: List[float] = []

    # The first value, 0, is used for padding to the length of the flow
    pdiff_up: List[float] = []
    pdiff_down: List[float] = []
    pdiff_global: List[float] = [0.]

    for idx, pkt in enumerate(curr_flow):
        if idx < pkt_num_threshold:
            pkts_global.append(len(pkt))

            if pkt.direction == PacketDirection.UPSTREAM:
                pkts_up.append(len(pkt))
                pkts_down.append(0.)
            else:
                pkts_down.append(len(pkt))
                pkts_up.append(0.)

    pdiff_up = get_inter_diff_by_stream(pkts_up)
    pdiff_down = get_inter_diff_by_stream(pkts_down)
    pdiff_global = get_inter_diff_by_stream(pkts_global)

    return pdiff_up, pdiff_down, pdiff_global


def get_sign_flag(curr_flow, pkt_num_threshold: int = 30) -> list:
    sign_flag_list = []

    for idx, pkt in enumerate(curr_flow):
        if idx < pkt_num_threshold:
            tcp_flags_value = pkt[TCP].flags.value
            sum_of_set_flags = TCPFlagNum.sum_of_set_flags(tcp_flags_value)
            sign_flag_list.append(
                sum_of_set_flags
                if pkt.direction == PacketDirection.UPSTREAM
                else -sum_of_set_flags
            )
        else:
            break

    return sign_flag_list
