import pytest

import numpy as np
import torch

from hypertor.aug.tcp.tor import TCPSequenceSimulator, TCPAugmentationConfig


@pytest.fixture
def dummy_data():
    """Create dummy TCP sequence data and labels"""
    base_features = np.array([
        [1, -1, 1, 1, 1, 1, -1], # pkt_dir_seq
        [1200, 1190, 1180, 1170, 1160, 1150, 1140], # pkt_len_seq
        [0, 115135, 115158, 115177, 115359, 233595, 233616], # pkt_usec_seq
        # ...
    ])
    # features = np.expand_dims(base_features, axis=0)
    features = np.stack([base_features, base_features+1000], axis=0)
    labels = np.array([0, 1])
    return features, labels


@pytest.fixture(params=[
    {
        "simulate_nagle": False,
        "simulate_packet_shift": False,
        "simulate_packet_loss": False,
        "simulate_packet_duplication": True,
        "max_rate": 0.5,
        "max_rtt": 0.01,
        "target_seq_len": 10,
    },
    {
        "simulate_nagle": False,
        "simulate_packet_shift": False,
        "simulate_packet_loss": True,
        "simulate_packet_duplication": False,
        "max_rate": 0.5,
        "max_rtt": 0.01,
        "target_seq_len": 10,
    },
    {
        "simulate_nagle": False,
        "simulate_packet_shift": True,
        "simulate_packet_loss": False,
        "simulate_packet_duplication": False,
        "max_rate": 0.5,
        "max_rtt": 0.01,
        "target_seq_len": 15,
    },
    {
        "simulate_nagle": True,
        "simulate_packet_shift": False,
        "simulate_packet_loss": False,
        "simulate_packet_duplication": False,
        "max_rate": 0.3,
        "max_rtt": 0.02,
        "target_seq_len": 20,
    },
    {
        "simulate_nagle": True,
        "simulate_packet_shift": True,
        "simulate_packet_loss": True,
        "simulate_packet_duplication": True,
        "max_rate": 0.3,
        "max_rtt": 0.02,
        "target_seq_len": 20,
    },
])
def simulator(dummy_data: tuple, request):
    features, labels = dummy_data
    """Create TCPSequenceSimulator instance with deterministic config"""
    cfg_kwargs = request.param
    # Fix seed for reproducibility
    config = TCPAugmentationConfig(seed=None, **cfg_kwargs)

    return TCPSequenceSimulator(config, features, labels)


def test_simulate(simulator: TCPSequenceSimulator):
    # simulator.data -> [num_samples, num_features, seq_len]
    res = simulator.augment_tcp_sequence(simulator.data[0])

    # This first packet in sequence is always keeped 
    assert res[0][0] == simulator.data[0][0][0]


def test_getitem_and_fit_seq(simulator: TCPSequenceSimulator):
    """Test __getitem__ and _fit_seq"""
    # input -> [samples, num_features, seq_len]
    # seq   -> [num_features, seq_len]
    # label -> [samples,]
    (seq_orig, seq_aug), label = simulator[0]

    print(f"seq_orig: {seq_orig}")
    print(f"seq_aug: {seq_aug}")
    print(f"simulator.aug_config: {simulator.aug_config}")

    # Types
    assert isinstance(seq_orig, torch.Tensor)
    assert isinstance(seq_aug, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    # Sequence length
    assert seq_orig.shape[1] == simulator.target_seq_len
    assert seq_aug.shape[1] == simulator.target_seq_len

    # Values are float tensors
    assert seq_orig.dtype == torch.float32
    assert seq_aug.dtype == torch.float32
