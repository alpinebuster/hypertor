from typing import List, TYPE_CHECKING
from pathlib import Path
from itertools import islice, zip_longest
import os
import glob
import re
import uuid
from enum import Enum, auto

from scapy.packet import Packet
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class PacketDirection(Enum):
    """Packet Direction creates an enumeration with the constant values \
    `UPSTREAM` and `DOWNSTREAM` for the direction of the packets.
    """
    UPSTREAM = auto()
    DOWNSTREAM = auto()


class TCPFlag(Enum):
    URG = "U"  # "URG": Urgent, indicates urgent data
    FIN = "F"  # "FIN": Finish, used to terminate a connection
    SYN = "S"  # "SYN": Synchronize, used to initiate a connection
    RST = "R"  # "RST": Reset, used to reset a connection
    PSH = "P"  # "PSH": Push, indicates that the data should be pushed to the receiving application
    ACK = "A"  # "ACK": Acknowledgment, confirms receipt of data
    ECE = "E"  # "ECE": ECN-Echo, used for congestion notification
    CER = "C"  # "CWR": Congestion Window Reduced, used in congestion control


class TCPFlagNum(Enum):
    FIN = int(0x01)
    SYN = int(0x02)
    RST = int(0x04)
    PSH = int(0x08)
    ACK = int(0x10)
    URG = int(0x20)
    ECE = int(0x40)
    CWR = int(0x80)

    @staticmethod
    def from_value(value):
        return [flag.value for flag in TCPFlagNum if value & flag.value]

    @staticmethod
    def flag_names_from_value(value):
        return [flag.name for flag in TCPFlagNum if value & flag.value]

    @staticmethod
    def sum_of_set_flags(value):
        return sum(flag.value for flag in TCPFlagNum if value & flag.value)


def get_flow_key(pkt: Packet, direction: PacketDirection) -> tuple:
    """Creates a unique id for a packet so it can be assigned to a flow.

    Args:
        packet (Packet): _description_
        direction (PacketDirection): _description_

    Returns:
        tuple: A tuple of the target IPv4 addresses.
    """

    if "TCP" in pkt:
        protocol = "TCP"
    else:
        raise Exception("Only TCP protocols are supported.")

    if pkt.haslayer("IP"):
        ip_lyr = pkt["IP"]
    else:
        ip_lyr = pkt["IPv6"]

    if direction == PacketDirection.UPSTREAM:
        dest_ip = ip_lyr.dst
        src_ip = ip_lyr.src
        src_port = pkt[protocol].sport
        dest_port = pkt[protocol].dport
    else:
        dest_ip = ip_lyr.src
        src_ip = ip_lyr.dst
        src_port = pkt[protocol].dport
        dest_port = pkt[protocol].sport

    return src_ip, dest_ip, src_port, dest_port


def mkdir(path: str) -> str:
    """If the directory does not exist, create it"""
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def grouper(iterable, chunk_len, max_groups=0, fillvalue=None):
    """Group data into fixed-length chunks.

    # Example data
    data = [1, 2, 3, 4, 5, 6, 7]
    # Group data into chunks of 3
    result = list(grouper(data, 3))
    print(result)  # Output: [(1, 2, 3), (4, 5, 6), (7, fillvalue, fillvalue)]

    Args:
        iterable (_type_): _description_
        n (_type_): each chunk's length
        max_groups (int, optional): _description_. Defaults to 0.
        fillvalue (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if max_groups > 0:
        iterable = islice(iterable, max_groups * chunk_len)

    # This takes the list created in the previous step and repeats it n times.
    # However, since lists in Python contain references to objects,
    # each element in this list is a reference to the same iterator object.
    args = [iter(iterable)] * chunk_len
    # `zip_longest` pulls one element from each iterator at a time.
    # Since all elements in iterators are the same iterator,
    # it pulls successive elements from the single iterator.
    return zip_longest(*args, fillvalue=fillvalue)


def random_string():
    return uuid.uuid4().hex[:6].upper().replace("0", "X").replace("O", "Y")


def get_statistics(alist: list):
    """Get summary statistics of a list"""
    iat = dict()
    alist = [float(x) for x in alist]

    if len(alist) > 1:
        iat["total"] = sum(alist)
        iat["max"] = max(alist)
        iat["min"] = min(alist)
        iat["mean"] = np.mean(alist)
        iat["std"] = np.sqrt(np.var(alist))
    else:
        iat["total"] = 0
        iat["max"] = 0
        iat["min"] = 0
        iat["mean"] = 0
        iat["std"] = 0

    return iat


def is_pcap_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in [".pcap", ".pcapng"]


def is_npy_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in [".npy"]


def is_csv_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in [".csv"]


def get_inter_diff_by_stream(arr: List[float], coefs: float = 1.0) -> List[float]:
    # >>> [1.2, 0, 1.8, 0, 0, 2.5]
    np_arr = np.array(arr)

    # >>> [0, 2, 5]
    nonzero_indices = np.nonzero(np_arr)[0]
    # Calculate the difference between adjacent non-zero elements
    # >>> [0.6, 0.7]
    diffs = np.diff(np_arr[nonzero_indices])

    result = np.zeros_like(np_arr, dtype=float)
    # Write the differences into the corresponding positions of the result array
    # >>> [0., 0., 0.6, 0., 0., 0.7]
    result[nonzero_indices[1:]] = coefs * diffs

    return result.tolist()


def replenish_list(_list: list, target_len: int = 30) -> list:
    (
        _list.extend([float("inf")] * (target_len - len(_list)))
        if len(_list) < target_len
        else _list
    )

    return _list
