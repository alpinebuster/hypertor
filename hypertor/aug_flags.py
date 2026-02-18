"""Aug flags."""

from absl import flags


# Data Augmentation
flags.DEFINE_bool(
    "simulate_nagle",
    False,
    "Enable simulation of the Nagle algorithm, which aggregates small packets to reduce transmission overhead."
)
# NOTE: Duplication already simulated in nagle
flags.DEFINE_bool(
    "simulate_packet_duplication",
    False,
    "Enable simulation of packet duplication to test handling of duplicated packets."
)
flags.DEFINE_bool(
    "simulate_packet_shift",
    False,
    "Enable simulation of packet reordering to evaluate system behavior under out-of-order delivery."
)
flags.DEFINE_bool(
    "simulate_packet_loss",
    False,
    "Enable simulation of packet loss to assess system robustness in unreliable network conditions."
)

flags.DEFINE_integer(
    "num_flows",
    1,
    "Number of flows.",
)
flags.DEFINE_integer(
    "num_pkt_features",
    1,
    "Number of packet feature types per flow.",
)
flags.DEFINE_integer(
    "tcp_seq_len",
    30,
    "Number of packets selected per TCP flow for sequence modeling.",
)
flags.DEFINE_integer(
    "drop_seq_len",
    3,
    "The minimal length for seq.",
)
