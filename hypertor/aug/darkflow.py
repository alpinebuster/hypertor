# DARK FLOW
# 
# The upstream and downstream behavior feature group,
# packet length distribution feature group,
# and extensible general traffic feature group (IAT, IPD, etc.).
#

import os
import ast

from absl import app, flags, logging
import numpy as np
import pandas as pd  # type:ignore

import hypertor.aug.tcp as tcp_aug
from hypertor.core import util
from hypertor import common_flags  # pylint:disable=unused-import
from hypertor import eval_model_flags  # pylint:disable=unused-import
from hypertor import variable_proto_flags  # pylint:disable=unused-import
from hypertor import aug_flags  # pylint:disable=unused-import

FLAGS = flags.FLAGS


def get_inter_diff_by_stream(arr: list[float], coefs: float = 1.0) -> list[float]:
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


def parse_list_column(s: str) -> list[int]:
    try:
        return ast.literal_eval(s)
    except:
        return []

def min_max_normalize_np(arr: np.ndarray):
    min_val = arr.min(axis=1, keepdims=True)
    max_val = arr.max(axis=1, keepdims=True)
    denom = max_val - min_val
    denom[denom == 0] = 1
    return (arr - min_val) / denom

def log_min_max_normalize_np(arr: np.ndarray):
    arr = np.log1p(arr)
    return min_max_normalize_np(arr)

def df_to_normalized_numpy(
    df: pd.DataFrame,
    target_len: int = 30,
    drop_len: int = 3,
) -> tuple[np.ndarray, list[str]]:
    # NOT perform norm on 'pkt_usec_seq' & 'pkt_len_seq' for data augmentation
    cols_small = ['pkt_dir_seq', 'pkt_len_seq', 'pkt_usec_seq', 
                  'fin_seq','syn_seq','rst_seq','psh_seq','ack_seq',
                  'urg_seq','cwr_seq','ece_seq','ip_df_seq','ip_mf_seq']
    cols_numeric = ['tcp_window_seq']
    cols_large = ['ttl_seq']

    all_cols = cols_small + cols_numeric + cols_large
    logging.info(f"len(df): {len(df)}")

    # Discard the rows with a sequence length less than `drop_len`
    seq_len = df['pkt_dir_seq'].apply(
        lambda x: len(parse_list_column(x))
    )
    df = df[seq_len>=drop_len].reset_index(drop=True)

    num_samples = len(df)
    num_features = len(all_cols)
    X = np.zeros((num_samples, num_features, target_len), dtype=np.float32)

    def _fit_list(_list: list, _target_len: int = 30) -> list:
        if len(_list) < _target_len:
            _list.extend([0] * (_target_len-len(_list)))
        else:
            _list = _list[:_target_len]
        return _list

    # Parse the columns at one time and stack them into numpy
    for j, col in enumerate(all_cols):
        # `apply`: call the specified function once for each element of the Series
        lists = df[col].apply(parse_list_column) # list of list
        lists = lists.apply(lambda l: _fit_list(l, target_len))

        X[:, j, :] = np.stack(lists.to_list(), axis=0)

    # Normalization
    #   1) cols_small -> remains unchanged
    #   2) cols_numeric -> min-max
    #   3) cols_large -> log + min-max
    idx_numeric = [all_cols.index(c) for c in cols_numeric]
    X[:, idx_numeric, :] = min_max_normalize_np(X[:, idx_numeric, :])

    idx_large = [all_cols.index(c) for c in cols_large]
    X[:, idx_large, :] = log_min_max_normalize_np(X[:, idx_large, :])

    return X, all_cols

def main(argv):
    """
    Run from `hypertor` project's root dir

    Input & Output:
       1) csv -> {
           "pkt_dir_seq": Vec<i32>, // Direction (/ -)
           "pkt_len_seq": Vec<i32>,
           "pkt_usec_seq": Vec<i64>, // Relative time us

           # ===== TCP flags =====
           "tcp_window_seq": Vec<u16>,

           "fin_seq": Vec<u8>,
           "syn_seq": Vec<u8>,
           "rst_seq": Vec<u8>,
           "psh_seq": Vec<u8>,
           "ack_seq": Vec<u8>,
           "urg_seq": Vec<u8>, // Rarely used
           "cwr_seq": Vec<u8>, // Rarely used
           "ece_seq": Vec<u8>, // Rarely used

           # ===== IP =====
           "ttl_seq": Vec<u8>,
           "ip_df_seq": Vec<u8>,
           "ip_mf_seq": Vec<u8>,
       }
        |
        | TCP Augmentation
        ↓ 
       2) npy [] -> {
           "SIGN": replenish_list(sign),
           "SIGN_PKT_LEN": replenish_list(sign_pkt_len),
           "SIGN_FLAG": replenish_list(sign_flag),
           "IAT_UP": replenish_list(iat_up),
           "IAT_DOWN": replenish_list(iat_down),
           "IAT_GLOBAL": replenish_list(iat_global),
           "INTER_PAYLOAD_DIFF_UP": replenish_list(inter_payload_diff_up),
           "INTER_PAYLOAD_DIFF_DOWN": replenish_list(inter_payload_diff_down),
           "INTER_PAYLOAD_DIFF_GLOBAL": replenish_list(inter_payload_diff_global),
       }
        |
        | Normalize
        ↓ 
       3) Convert to a 5D tensor [all_classes, all_samples, F, C, T]
    """
    # The command-line parameters have been parsed by absl.
    if len(argv) > 1:
        del argv

    gpus: str = FLAGS.gpus
    if gpus is None or gpus == "all":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    util.print_gpu_detailed_info()

    df = pd.read_csv('./data/newrealtime2.csv', encoding='utf-8-sig')
    logging.info(df.head())

    config = tcp_aug.make_tcp_augmentation_config()

    # TODO
    labels = np.zeros(df.shape[0])
    simulator = tcp_aug.TCPSequenceSimulator(config, df, labels)
    # df=simulator.data -> [num_samples, num_features, seq_len] or [all_classes, all_samples, F, C, T]
    res = simulator.augment_tcp_sequence(simulator.data[0])

    # X -> [samples, H, W]
    X, col_order = df_to_normalized_numpy(df, target_len=config.target_seq_len)
    logging.info(f"X.shape: {X.shape}") # (num_samples, num_features, seq_len)
    logging.info(f"Columns: {col_order}")


if __name__ == "__main__":
    app.run(main)
