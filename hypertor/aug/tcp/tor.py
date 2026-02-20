import random
import dataclasses
import os
import functools

from typing import Optional

from absl import app, flags, logging
import pandas as pd
import numpy as np
import torch

FLAGS = flags.FLAGS


@dataclasses.dataclass
class TCPAugmentationConfig:
    """TCP augmentation configuration."""

    simulate_nagle: bool = False
    simulate_packet_shift: bool = False
    simulate_packet_loss: bool = False
    simulate_packet_duplication: bool = False

    # random.random() > random.random()*max_rate
    # X ~ Uniform(a, b) -> E[X]=(a+b)/2 -> E[X]=E[X]/2
    max_rate: float = 0.404
    max_rtt: float = 0.011 # seconds
    mss: int = 1460
    seed: Optional[int] = None
    anomaly_delay: float = 0.3 # 300 ms, normal RTT: 10-50 ms, peak RTT: 150-300+ ms
    loc: float = 6e-06 # 6 μs
    scale: float = 0.012 # 12 ms
    target_seq_len: int = 100
    drop_seq_len: int = 3
    num_flows: int = 1
    num_pkt_features: int = 3

def make_tcp_augmentation_config():
    return TCPAugmentationConfig(
        simulate_nagle=FLAGS.simulate_nagle,
        simulate_packet_shift=FLAGS.simulate_packet_shift,
        simulate_packet_loss=FLAGS.simulate_packet_loss,
        simulate_packet_duplication=FLAGS.simulate_packet_duplication,
        target_seq_len=FLAGS.tcp_seq_len,
        drop_seq_len=FLAGS.drop_seq_len,
        num_flows=FLAGS.num_flows,
        num_pkt_features=FLAGS.num_pkt_features,
    )


class TCPSequenceSimulator:
    """
    1. Nagle, aggregating small packets

    2. Packet Duplication (TCP Retransmission/Fragmentation)
       2.1 RTO (Retransmission Timeout) 
        → t_retx = t_original + RTO, originally
        →1 large packet → divided into 2-3 small packets  
       2.2 Fast Retransmission
        → t_retx = t_original + 3×mean_iat
    
    3. Packet Sequence Shift (Out-of-Order Arrival)
    4. Packet Loss
    """

    def __init__(
        self,
        aug_config: TCPAugmentationConfig,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ):
        self.aug_config = aug_config

        if self.aug_config.seed is not None:
            seed = self.aug_config.seed

            # Python random
            random.seed(seed)
            # NumPy random
            np.random.seed(seed)
            # Torch CPU
            torch.manual_seed(seed)

            # Torch CUDA
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # TCP: Round Trip Time
        self.max_rtt = self.aug_config.max_rtt
        # TCP: Maximum Segment Size
        # MSS = ETHER_MTU - IP_HEADER - TCP_HEADER 
        #     = 1500 - 20 - 20
        #     = 1460 bytes
        self.mss = self.aug_config.mss

        self.max_rate = self.aug_config.max_rate
        self.anomaly_delay = self.aug_config.anomaly_delay
        self.loc = self.aug_config.loc
        self.scale = self.aug_config.scale
        self.target_seq_len = self.aug_config.target_seq_len

        if labels is not None:
            self.len = len(labels)
        self.labels = labels # [num_samples,]
        self.data = data # [num_samples, num_features, seq_len] or [all_classes, all_samples, F, C, T]

        self.delays: Optional[list[float]] = None
        self.relative_send_times: Optional[list[float]] = None

    def random_delay_distribution(self, seq_length: int) -> tuple[list[float], list[float]]:
        delays: list[float] = []

        while len(delays) < seq_length:
            t = random.random() # Uniform distribution: [0.0, 1.0) in seconds

            if t < 0.1:
                delays.append(self.anomaly_delay)
            else:
                # Exponential Distribution
                # 
                # 1. `loc` determines the minimum delay (the starting point of the distribution)
                #   * Ensures that all generated delays are at least above a minimum value
                # 2. `scale` determines the average delay and its variability (the width of the distribution). 
                #   * Larger scale  → larger delay variations (network congestion / higher jitter)
                #      t|
                #       |  \
                #       |   \
                #       |    \
                #       |     \__________
                #       +----------------> x
                #   * Smaller scale → smaller delay variations (a more stable network)
                #      t|
                #       | \
                #       |  \
                #       |   \____________
                #       +----------------> x
                #
                delays.append(np.random.exponential(scale=self.scale) + self.loc)
        relative_send_times = np.cumsum(delays).tolist()

        return delays, relative_send_times

    def simulate_nagle(self, in_seq: np.ndarray, mss: int) -> np.ndarray:
        """
        Simulate Nagle's algorithm by aggregating small packets into larger ones based on RTT and MSS.

        :param in_seq: e.g., [1300, 1400, 500, 1460, 1460, 1200]
        :param mss: e.g., 1460
        :return: e.g., [1300, 1400, 500, 1460, 1460, 1200] or [300, 1900, 1460, 1460, 1200], etc.
        """
        seq_t: list[list] = in_seq.T.tolist()

        res: list[list[int]] = []
        # NOTE: `seq_t.pop(0)` will remove the first element from the original list
        # Directly add the first two packages to the result,
        # Nagle usually allows the first package to be sent directly without waiting for merging
        res.append(seq_t.pop(0))
        res.append(seq_t.pop(0))

        # In seconds
        self.delays, self.relative_send_times = self.random_delay_distribution(len(seq_t))
        assert self.delays is not None
        delay_iter = iter(self.delays) 
        time_offset_usec = 0
        while len(seq_t) > 0:
            cur_rtt = random.random() * self.max_rtt
            buf = 0
            buf_pkt: list = []

            while len(seq_t) > 0 and cur_rtt > 0:
                # delay = self.delays.pop(0)
                delay = next(delay_iter)

                cur_rtt -= delay

                pkt = seq_t.pop(0)  # full feature vector
                # features: [pkt_dir_seq,pkt_len_seq,pkt_usec_seq,tcp_window_seq,fin_seq,syn_seq,rst_seq,psh_seq,ack_seq,urg_seq,cwr_seq,ece_seq,ttl_seq,ip_df_seq,ip_mf_seq]
                pkt_len = pkt[1]
                pkt_time_us = int(pkt[2])

                buf += int(pkt_len)
                buf_pkt = pkt.copy()

                # If buffer exceeds MSS, send in chunks
                while buf > mss:
                    buf -= mss

                    # Send MSS-sized packet
                    to_send = pkt.copy() # copy packet's features
                    to_send[1] = mss  # set length to MSS

                    # Simulate the sending delay of duplication
                    to_send[2] = pkt_time_us + time_offset_usec
                    dup_delay, _ = self.random_delay_distribution(seq_length=1)
                    dup_delay_usec = int(dup_delay[0] * 1_000_000)
                    time_offset_usec += dup_delay_usec

                    res.append(to_send)

                buf_pkt[2] = pkt_time_us + time_offset_usec

            if buf_pkt is not None and len(buf_pkt) >= 3:
                buf_pkt[1] = buf
                res.append(buf_pkt)

        return np.array(res).T

    def simulate_packet_loss(self, in_seq: np.ndarray) -> np.ndarray:
        """
        Simulate packet loss by randomly dropping packets based on a loss rate.
        The first packet is always kept as is. The loss rate is randomly chosen up to a maximum value.

        :param in_seq: e.g., [1, 2, 3, 4, 5]
        :return: e.g., [1, 3, 5], [1, 2, 4] or [1, 2, 3, 4, 5], etc.
        """
        seq_t: list[list] = in_seq.T.tolist()

        res: list[list] = []
        # Always keep the first packet
        res.append(seq_t.pop(0))

        # NOTE: (self.max_rate/2)
        loss_rate = random.random() * (self.max_rate)
        for pkt in seq_t:
            if random.random() > loss_rate:
                # Packet survives
                res.append(pkt)

        return np.array(res).T

    def simulate_packet_duplication(self, in_seq: np.ndarray) -> np.ndarray:
        """
        Simulate packet duplication by randomly duplicating packets based on a loss rate.

        :param in_seq: e.g., [1, 2, 3, 4, 5]
        :return: e.g., [1, 2, 2, 3, 4, 4, 5], [1, 1, 2, 3, 3, 4, 5, 5] or [1, 2, 3, 4, 5, 5], etc.
        """
        seq_t: list[list] = in_seq.T.tolist()

        res: list[list] = []
        # Always keep the first packet
        res.append(seq_t.pop(0))

        max_dup = 3
        duplication_rate = random.random() * self.max_rate
        time_offset_usec = 0
        for pkt in seq_t:
            pkt_time_us = int(pkt[2])

            base_pkt = pkt.copy()
            # The offset brought by possible previous duplication should be added
            base_pkt[2] = pkt_time_us + time_offset_usec
            res.append(base_pkt)

            # Randomly duplicate packets based on loss rate
            # It may be repeated multiple times
            dup_count = 0
            cur_rate = duplication_rate
            while dup_count < max_dup and random.random() > cur_rate:
                # Simulate the sending delay of duplication
                dup_delay, _ = self.random_delay_distribution(seq_length=1)
                dup_delay_usec = int(dup_delay[0] * 1_000_000)

                dup_pkt = pkt.copy()
                time_offset_usec += dup_delay_usec
                dup_pkt[2] = pkt_time_us + time_offset_usec

                res.append(dup_pkt)

                dup_count += 1
                cur_rate *= .5

        return np.array(res).T

    def simulate_packet_shift(self, in_seq: np.ndarray) -> np.ndarray:
        """
        Perform subsequence shifting by randomly dropping packets to the new sequence based on a loss rate.

        :param in_seq: [num_features, seq_len] -> Example feature: [1, 2, 3, 4, 5]
        :return: e.g., [1, 3, 5, 2, 4], [1, 2, 4, 3, 5] or [1, 3, 2, 4, 5], etc.
        """
        def _generate_shifted_indices(max_rate, seq_len: int) -> list[int]:
            indices = list(range(seq_len))

            res = [indices.pop(0)]  # always keep first
            shift_rate = random.random()*max_rate

            while indices:
                for i in range(len(indices)):
                    if random.random() > shift_rate:
                        res.append(indices.pop(i))
                        break
            return res

        num_features, seq_len = in_seq.shape
        # Generate shifted packet sequence indices
        new_order = _generate_shifted_indices(self.max_rate, seq_len)

        # [num_features, seq_len]
        out = np.empty_like(in_seq)
        # features: [pkt_dir_seq,pkt_len_seq,pkt_usec_seq,tcp_window_seq,fin_seq,syn_seq,rst_seq,psh_seq,ack_seq,urg_seq,cwr_seq,ece_seq,ttl_seq,ip_df_seq,ip_mf_seq]
        time_idx = 2
        # Maintain the original order for `pkt_usec_seq`
        out[time_idx] = in_seq[time_idx]

        # Reorder by new_order for other features
        for f in range(num_features):
            if f == time_idx:
                continue
            out[f] = in_seq[f, new_order]

        return out

    def augment_tcp_sequence(self, seq: np.ndarray) -> np.ndarray:
        # seq -> [num_features, seq_len] or [C, T]
        if seq is None or seq.size == 0:
            raise ValueError("Input sequence is empty or None")
        _data = seq.copy()

        # The priority order is: 
        #    → RTO timeout retransmission
        #    → Fast Retransmit
        #    → normal sending (subject to Nagle restrictions)
        if self.aug_config.simulate_nagle:
            _data = self.simulate_nagle(_data, self.mss)

        # Build a list of tuples: (method_function, enabled_flag)
        ops = [
            (self.simulate_packet_shift, self.aug_config.simulate_packet_shift),
            (self.simulate_packet_loss, self.aug_config.simulate_packet_loss),
            (self.simulate_packet_duplication, self.aug_config.simulate_packet_duplication),
        ]
        # Keep only enabled operations
        enabled_ops = [
            func
            for func, enabled in ops
            if enabled
        ]
        # Shuffle order
        random.shuffle(enabled_ops)
        # Apply in random order
        for func in enabled_ops:
            _data = func(_data)

        return self._fit_seq(np.array(_data, dtype=np.float32))

    def _fit_seq(self, seq_x: np.ndarray) -> np.ndarray:
        # [num_features, seq_len]
        _, current_seq_len = seq_x.shape

        if current_seq_len > self.target_seq_len:
            # If it's too long, cut it out first
            seq_x = seq_x[:, :self.target_seq_len]
        else:
            # Pad to the specified length
            pad_width = ((0, 0), (0, self.target_seq_len-current_seq_len))
            seq_x = np.pad(seq_x, pad_width=pad_width, mode='constant', constant_values=0)

        return seq_x


if __name__ == "__main__":
    base_features = np.array([
        [1, -1, 1, 1, 1, 1, -1], # pkt_dir_seq
        [1200, 1190, 1180, 1170, 1160, 1150, 1140], # pkt_len_seq
        [0, 115135, 115158, 115177, 115359, 233595, 233616], # pkt_usec_seq
        # ...
    ])
    features = np.stack([base_features, base_features+1000], axis=0)
    args = {
        "simulate_nagle": False,
        "simulate_packet_shift": True,
        "simulate_packet_loss": False,
        # "simulate_packet_duplication": True,
        "max_rate": 0.404,
        "max_rtt": 0.011,
        "target_seq_len": 30,
    }
    config = TCPAugmentationConfig(seed=None, **args)
    simulator = TCPSequenceSimulator(config, features)
    assert simulator.data is not None
    res = simulator.augment_tcp_sequence(simulator.data[0])
    print(f"simulator.data[0]: {simulator.data[0]}")
    print(f"res: {res}")
