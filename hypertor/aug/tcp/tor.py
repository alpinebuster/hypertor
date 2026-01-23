import random
import dataclasses
import os
import functools

from typing import Any, Optional

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

    max_rate: float = 0.704
    max_rtt: float = 0.011
    mss: int = 1460
    seed: Optional[int] = None
    anomaly_delay: float = 0.3 # 300 ms, normal RTT: 10-50 ms, peak RTT: 150-300+ ms
    loc: float = 6e-06 # 6 μs
    scale: float = 0.012 # 12 ms
    seq_len: int = 100

def make_tcp_augmentation_config():
    return TCPAugmentationConfig(
        simulate_nagle=FLAGS.simulate_nagle,
        simulate_packet_shift=FLAGS.simulate_packet_shift,
        simulate_packet_loss=FLAGS.simulate_packet_loss,
        simulate_packet_duplication=FLAGS.simulate_packet_duplication,
        seq_len=FLAGS.tcp_seq_len,
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
        data: np.ndarray,
        labels: np.ndarray,
    ):
        self.aug_config = aug_config

        if self.aug_config.seed is not None:
            random.seed(self.aug_config.seed)

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
        self.seq_len = self.aug_config.seq_len

        self.len = len(labels)
        self.labels = labels # [num_samples,]
        self.data = data # [num_samples, num_features, seq_len]

        self.delays: Optional[list[float]] = None
        self.relative_send_times: Optional[list[float]] = None

    def random_delay_distribution(self, seq_length: int) -> tuple[list[float], list[float]]:
        delays: list[float] = []

        while len(delays) < seq_length:
            t = random.random() # Uniform distribution: [0.0, 1.0)

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

    def simulate_nagle(self, in_seq: list[int], mss: int) -> list[int]:
        """
        Simulate Nagle's algorithm by aggregating small packets into larger ones based on RTT and MSS.

        :param in_seq: e.g., [1300, 1400, 500, 1460, 1460, 1200]
        :param mss: e.g., 1460
        :return: e.g., [1300, 1400, 500, 1460, 1460, 1200] or [300, 1900, 1460, 1460, 1200], etc.
        """
        res: list[int] = []
        # NOTE: `in_seq.pop(0)` will remove the first element from the original list
        # Directly add the first two packages to the result,
        # Nagle usually allows the first package to be sent directly without waiting for merging
        res.append(in_seq.pop(0))
        res.append(in_seq.pop(0))

        self.delays, self.relative_send_times = self.random_delay_distribution(len(in_seq))
        assert self.delays is not None
        while len(in_seq) > 0:
            cur_rtt = random.random() * self.max_rtt
            buf = 0
            while len(in_seq) > 0 and cur_rtt > 0:
                delay = self.delays.pop(0)
                cur_rtt -= delay
                buf += int(in_seq.pop(0))
                while buf > mss:
                    buf -= mss
                    res.append(mss)
            res.append(buf)

        return res

    def simulate_packet_loss(self, in_seq: list) -> list:
        """
        Simulate packet loss by randomly dropping packets based on a loss rate.
        The first packet is always kept as is. The loss rate is randomly chosen up to a maximum value.

        :param in_seq: e.g., [1, 2, 3, 4, 5]
        :return: e.g., [1, 3, 5], [1, 2, 4] or [1, 2, 3, 4, 5], etc.
        """
        res: list[Any] = []

        if not in_seq:
            return res

        # Always keep the first packet
        res.append(in_seq.pop(0))

        loss_rate = random.random() * self.max_rate
        for pkt in in_seq:
            if random.random() > loss_rate:
                # Packet survives
                res.append(pkt)

        return res

    def simulate_packet_duplication(self, in_seq: list) -> list:
        """
        Simulate packet duplication by randomly duplicating packets based on a loss rate.

        :param in_seq: e.g., [1, 2, 3, 4, 5]
        :return: e.g., [1, 2, 2, 3, 4, 4, 5], [1, 1, 2, 3, 3, 4, 5, 5] or [1, 2, 3, 4, 5, 5], etc.
        """
        res: list[Any] = []

        if not in_seq:
            return res

        # Always keep the first packet
        res.append(in_seq.pop(0))

        loss_rate = random.random() * self.max_rate
        for pkt in in_seq:
            res.append(pkt)

            # FIXME: Randomly duplicate packets based on loss rate
            # It may be repeated multiple times
            while random.random() > loss_rate:
                res.append(pkt)

        return res

    def simulate_packet_shift(self, in_seq: list) -> list:
        """
        Perform subsequence shifting by randomly dropping packets to the new sequence based on a loss rate.

        :param in_seq: e.g., [1, 2, 3, 4, 5]
        :return: e.g., [1, 3, 5, 2, 4], [1, 2, 4, 3, 5] or [1, 3, 2, 4, 5], etc.
        """
        res: list[Any] = []
        # Always keep the first packet
        res.append(in_seq.pop(0))

        loss_rate = random.random() * self.max_rate
        while len(in_seq) > 0:
            for i in range(len(in_seq)):
                if random.random() > loss_rate:
                    res.append(in_seq.pop(i))
                    break

        return res

    def augment_tcp_sequence(self, seq: np.ndarray) -> np.ndarray:
        data = seq.tolist()

        # Build a list of tuples: (method_function, enabled_flag)
        ops = [
            (functools.partial(self.simulate_nagle, mss=self.mss), self.aug_config.simulate_nagle),
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
            data = func(data)

        return np.array(data)


    def _fit_seq(self, seq_x: np.ndarray) -> torch.Tensor:
        # If it's too long, cut it out first
        seq_x = seq_x[:self.seq_len]
        # Pad to the specified length
        seq_x = np.pad(
            seq_x,
            pad_width=(0, max(0, self.seq_len - len(seq_x))),
            mode='constant',
        )
        return torch.from_numpy(seq_x).float()

    def __getitem__(self, index: int) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        seq = self.data[index]
        seq_auged = self.augment_tcp_sequence(seq)

        seq = self._fit_seq(seq)
        seq_auged = self._fit_seq(seq_auged)
        ground_truth = torch.tensor(self.labels[index], dtype=torch.long)
        return (seq, seq_auged), ground_truth

    def __len__(self) -> int:
        return len(self.data)
