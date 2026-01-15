import random
from typing import Any, Optional, Tuple

import numpy as np
from scipy.stats import expon # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import torch


class TCPSequenceSimulator:
    def __init__(
        self,
        data,
        max_rate: float = 0.3,
        max_rtt: float = 0.01,
        enable_nagle: bool = True,
        mss: int = 1460,
        seed: Optional[int] = None,
        anomaly_delay: float = 0.3, # 300 ms, normal RTT: 10-50 ms, peak RTT: 150-300+ ms
        loc: float = 6e-06, # 6 μs
        scale: float = 0.012, # 12 ms
        seq_len: int = 100,
        patch_size: int = 10,
    ):
        if seed is not None:
            random.seed(seed)

        # TCP: Round Trip Time
        self.max_rtt = max_rtt
        # TCP: Nagle Algorithm
        self.enable_nagle = enable_nagle
        # TCP: Maximum Segment Size
        # MSS = ETHER_MTU - IP_HEADER - TCP_HEADER 
        #     = 1500 - 20 - 20
        #     = 1460 bytes
        self.mss = mss

        self.max_rate = max_rate
        self.anomaly_delay = anomaly_delay
        self.loc = loc
        self.scale = scale
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.label_encoder = LabelEncoder()

        labels = np.array(data['labels'])
        self.len = len(labels)
        self.data = np.array(data.drop('labels', axis=1))
        self.label = self._encode_labels(labels)

    def _encode_labels(self, seq_y):
        self.label_encoder.fit(list(set(seq_y)))
        seq_y = self.label_encoder.transform(seq_y)
        return torch.from_numpy(seq_y)

    def random_delay_distribution(self, seq_length: int) -> list:
        delays = []

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
                t = expon.rvs(loc = self.loc, scale = self.scale)
                t = np.atleast_1d(t)
                delays.extend(t)

        return delays

    def _fit_seq_length(self, seq, tl):
        label = seq[-1]
        data = seq[:-1]
        if len(data) < tl:
            while len(data) < tl:
                data.append(0)
        else:
            data = seq[:tl]

        data.append(label)
        return data

    def simulate_nagle(self, in_seq: list, mss: int) -> list:
        """
        Simulate Nagle's algorithm by aggregating small packets into larger ones based on RTT and MSS.

        :param in_seq: e.g., [3, 4, 5, 1460, 1460, 1200]
        :param mss: e.g., 1460
        :return: e.g., [3, 4, 5, 1460, 1460, 1200] or [3, 9, 1460, 1460, 1200], etc.
        """
        res: list[Any] = []
        res.append(in_seq.pop(0))
        res.append(in_seq.pop(0))
        delays = self.random_delay_distribution(len(in_seq))

        while len(in_seq) > 0:
            cur_rtt = random.random() * self.max_rtt
            buf = 0

            while len(in_seq) > 0 and cur_rtt > 0:
                delay = delays.pop(0)
                cur_rtt -= delay
                buf += int(in_seq.pop(0))

                while buf > mss:
                    buf -= mss
                    res.append(mss)

            res.append(buf)

        res = self._fit_seq_length(res, 100)
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

            # Randomly duplicate packets based on loss rate
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

        if self.enable_nagle:
            data = self.simulate_nagle(data, self.mss)

        data = self.simulate_packet_shift(data)
        data = self.simulate_packet_loss(data)
        data = self.simulate_packet_duplication(data)

        return np.array(data)


    def _prepare_sequence(self, seq_x):
        # If it's too long, cut it out first
        seq_x = seq_x[:self.seq_len]
        # Pad to the specified length
        seq_x = np.pad(seq_x, (0, max(0, self.seq_len - len(seq_x))), 'constant')
        # Convert to patches
        seq_x = seq_x.reshape(-1, self.patch_size, self.patch_size)
        return torch.from_numpy(seq_x).float()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        seq = self.data[index]
        seq_auged = self.augment_tcp_sequence(seq)

        seq = self._prepare_sequence(seq)
        seq_auged = self._prepare_sequence(seq_auged)
        ground_truth = self.label[index]

        return (seq, seq_auged), ground_truth

    def __len__(self):
        return len(self.data)
