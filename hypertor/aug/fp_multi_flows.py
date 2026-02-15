# %%
import os
import ast
import glob
import ipaddress

import numpy as np
import pandas as pd

import hypertor.aug.tcp as tcp_aug
from hypertor.core import util
from hypertor.aug import darkflow

# %% [markdown]
# 1) Single Flow
#    * 30 -> raw, aug(nagle, shift, loss, all)
#    * 60 -> raw, aug(nagle, shift, loss, all)
#    * 90 -> raw, aug(nagle, shift, loss, all)
# 2) 3 Flow
#    ...
# 3) 6 Flow
#    ...
# 4) 9 Flow
#    ...

# %%
hypertor_data_path = "../../data/"

def normalize_flow_key(src_ip, src_port, dst_ip, dst_port):
    ip1 = int(ipaddress.ip_address(src_ip))
    ip2 = int(ipaddress.ip_address(dst_ip))

    p1 = int(src_port)
    p2 = int(dst_port)

    if (ip1, p1) <= (ip2, p2):
        return f"{ip1}:{p1}-{ip2}:{p2}"
    else:
        return f"{ip2}:{p2}-{ip1}:{p1}"


# %%
def _build(related_df, all_cols, target_len):
    C = len(all_cols)
    num_samples = len(related_df)

    X = np.zeros((num_samples, C, target_len), dtype=np.float32)

    def _fit_list(lst, target_len):
        if len(lst) < target_len:
            lst = lst + [0] * (target_len - len(lst))
        else:
            lst = lst[:target_len]
        return lst

    for j, col in enumerate(all_cols):
        lists = related_df[col].apply(darkflow.parse_list_column)
        lists = lists.apply(lambda l: _fit_list(l, target_len))
        X[:, j, :] = np.stack(lists.to_list(), axis=0)

    return X  # [F, C, T]

def _find_related_flows(
    aug_config: "tcp_aug.TCPAugmentationConfig",
    fp_df,
    row,
):
    base_df = pd.DataFrame([row._asdict()])
    def _attach_and_trim(candidate_df):
        candidate_df = candidate_df[
            candidate_df["norm_flow_key"] != row.norm_flow_key
        ]
        _combined = pd.concat([base_df, candidate_df])
        _combined = _combined.drop_duplicates()
        _combined = _combined.head(aug_config.num_flows)
        return _combined.reset_index(drop=True)

    # import pdb
    # pdb.set_trace()
    # print(f"ja4: {row.ja4}, ja4t: {row.ja4t}, ja4l: {row.ja4l}")
    # print(f"\nja4t: {fp_df[fp_df["ja4t"] == row.ja4t].shape}")
    # print(f"\nja4: {fp_df[fp_df["ja4"] == row.ja4].shape}")
    # print(f"\nja4_1: {fp_df[fp_df["ja4_1"] == row.ja4_1].shape}")
    # print(f"\nja4_3: {fp_df[fp_df["ja4_3"] == row.ja4_3].shape}")
    # print(f"\nja4l: {fp_df[fp_df["ja4l"] == row.ja4l].shape}")
    # print(f"\nja4l_1: {fp_df[fp_df["ja4l_1"] == row.ja4l_1].shape}")
    # print(f"\nja4l_2: {fp_df[fp_df["ja4l_2"] == row.ja4l_2].shape}")

    # print(f"\nja4l_1: {fp_df[fp_df["ja4l_1"] == row.ja4l_1]["ja4l"]}")
    # print(f"\nja4l_2: {fp_df[fp_df["ja4l_2"] == row.ja4l_2]["ja4l"]}")

    # -------- Level 1 --------
    level1 = fp_df[
        (fp_df["ja4t"] == row.ja4t) &
        (fp_df["ja4"] == row.ja4) &
        (fp_df["ja4l"] == row.ja4l)
    ]
    if len(level1) >= aug_config.num_flows:
        return _attach_and_trim(level1)
    # print(f"level1: {level1.shape}")

    # -------- Level 2 --------
    level2 = fp_df[
        (fp_df["ja4t"] == row.ja4t) &
        (fp_df["ja4_1"] == row.ja4_1) &
        (fp_df["ja4_3"] == row.ja4_3) &
        (fp_df["ja4l_2"] == row.ja4l_2)
    ].copy()
    # print(f"level2: {level2.shape}")

    if len(level2) > 0:
        if pd.isna(row.ja4l_1):
            level2["ja4l_1_diff"] = 1e9
        else:
            level2["ja4l_1_diff"] = (
                pd.to_numeric(level2["ja4l_1"], errors="coerce") - row.ja4l_1
            ).abs().fillna(1e9)
        level2 = level2.sort_values("ja4l_1_diff")

    if len(level2) >= aug_config.num_flows:
        return _attach_and_trim(level2)

    # -------- Level 3 --------
    level3 = fp_df[
        (fp_df["ja4t"] == row.ja4t) &
        (fp_df["ja4_1"] == row.ja4_1) &
        (fp_df["ja4_3"] == row.ja4_3)
    ].copy()
    # print(f"level3: {level3.shape}")

    if len(level3) > 0:
        if pd.isna(row.ja4l_2):
            level3["ja4l_2_match"] = False
        else:
            level3["ja4l_2_match"] = level3["ja4l_2"] == row.ja4l_2

        if pd.isna(row.ja4l_1):
            level3["ja4l_1_diff"] = 1e9
        else:
            level3["ja4l_1_diff"] = (
                pd.to_numeric(level3["ja4l_1"], errors="coerce") - row.ja4l_1
            ).abs().fillna(1e9)

        level3 = level3.sort_values(
            by=["ja4l_2_match", "ja4l_1_diff"],
            ascending=[False, True],
        )

    if len(level3) >= aug_config.num_flows:
        return _attach_and_trim(level3)
    
    # -------- Level 4 --------
    level4 = fp_df[
        (fp_df["ja4t"] == row.ja4t) &
        ((fp_df["ja4_1"] == row.ja4_1) | (fp_df["ja4_3"] == row.ja4_3))
    ].copy()
    # print(f"level4: {level4.shape}")

    if len(level4) > 0:
        if pd.isna(row.ja4l_1):
            level4["ja4l_1_diff"] = 1e9
        else:
            level4["ja4l_1_diff"] = (
                pd.to_numeric(level4["ja4l_1"], errors="coerce") - row.ja4l_1
            ).abs().fillna(1e9)
        level4 = level4.sort_values("ja4l_1_diff")

    if len(level4) >= aug_config.num_flows:
        return _attach_and_trim(level4)

    # -------- Level 5 --------
    combined = pd.concat([base_df, level4])
    combined = combined.drop_duplicates()

    # Padding
    pad_count = aug_config.num_flows - len(combined)
    if pad_count > 0:
        pad_df = pd.DataFrame([row._asdict()]*pad_count)
        combined = pd.concat([combined, pad_df])
    return combined.head(aug_config.num_flows).reset_index(drop=True)

def build_flows(
    row,
    fp_df,
    all_cols,
    aug_config: "tcp_aug.TCPAugmentationConfig",
):
    if aug_config.num_flows > 1:
        related_df = _find_related_flows(aug_config, fp_df, row)
    else:
        related_df = pd.DataFrame([row])  # To DataFrame
    # print(f"related_df: {related_df}")

    X = _build(
        related_df,
        all_cols,
        aug_config.target_seq_len
    )  # [F, C, T]

    F = aug_config.num_flows
    if X.shape[0] < F:
        pad = np.zeros(
            (F - X.shape[0], X.shape[1], X.shape[2]),
            dtype=np.float32,
        )
        X = np.concatenate([X, pad], axis=0)
    if X.shape[0] > F:
        X = X[:F]

    return X  # [F, C, T]

def build_dataset(
    df,
    fp_df,
    all_cols,
    aug_config: "tcp_aug.TCPAugmentationConfig",
):
    classes: list = sorted(df.label.unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"class_to_idx: {class_to_idx}")

    # [all_classes, samples_per_class, F, C, T]
    all_data: list[list] = [[] for _ in classes]
    for i, row in enumerate(df.itertuples(index=False), 1):
        print(f"[{i}/{len(df)}] label={row.label}, flow_key={row.flow_key}")
        label = row.label
        cls_idx = class_to_idx[label]

        # X shape = [F, C, T]
        X = build_flows(row, fp_df, all_cols, aug_config)
        # print(f"X: {X.shape}")
        all_data[cls_idx].append(X)

    F = aug_config.num_flows
    C = len(all_cols)
    T = aug_config.target_seq_len

    all_data_np = [
        np.stack(class_samples, axis=0)  # shape = [num_samples, F, C, T]
        for class_samples in all_data
    ]
    min_samples = min(d.shape[0] for d in all_data_np)
    dataset = np.zeros((len(classes), min_samples, F, C, T), dtype=np.float32)

    for cls_idx, d in enumerate(all_data_np):
        dataset[cls_idx, :d.shape[0]] = d

    return dataset

def csv2numpy(
    aug_config: "tcp_aug.TCPAugmentationConfig",
    dataset: str = "cicdarknet",
):
    if dataset.lower() == "cicdarknet":
        base_path = "../../data/cicdarknet/csv"
        lpbase_path = "../../data/lpcicdarknet/csv"
    elif dataset.lower() == "tcub":
        base_path = "../../data/tcub/csv"
        lpbase_path = "../../data/lptcub/csv"
    else:
        raise RuntimeError("Wrong dataset")

    base_file = os.path.join(base_path, "darkflow_raw_ja4cleaned.csv")
    fpbase_file = os.path.join(lpbase_path, "darkflow_raw.csv")

    df = pd.read_csv(base_file, encoding="utf-8-sig")
    fp_df = pd.read_csv(fpbase_file, encoding="utf-8-sig")


    fp_df[["ja4_1", "ja4_2", "ja4_3"]] = fp_df["ja4"].str.split("_", expand=True)
    for c in ["ja4_1", "ja4_2", "ja4_3"]:
        fp_df[c] = fp_df[c].fillna("").astype(str)

    fp_df[["ja4l_1", "ja4l_2"]] = fp_df["ja4l"].str.split("_", expand=True)
    fp_df["ja4l_1"] = pd.to_numeric(fp_df["ja4l_1"], errors="coerce").astype("Int64")
    fp_df["ja4l_2"] = pd.to_numeric(fp_df["ja4l_2"], errors="coerce").astype("Int64")


    df[["ja4_1", "ja4_2", "ja4_3"]] = df["ja4"].str.split("_", expand=True)
    for c in ["ja4_1", "ja4_2", "ja4_3"]:
        df[c] = df[c].fillna("").astype(str)

    df[["ja4l_1", "ja4l_2"]] = df["ja4l"].str.split("_", expand=True)
    df["ja4l_1"] = pd.to_numeric(df["ja4l_1"], errors="coerce").astype("Int64")
    df["ja4l_2"] = pd.to_numeric(df["ja4l_2"], errors="coerce").astype("Int64")


    simulator = tcp_aug.TCPSequenceSimulator(aug_config)
    cols_small = [
        'pkt_dir_seq', 'pkt_len_seq', 'pkt_usec_seq',
        'fin_seq','syn_seq','rst_seq','psh_seq','ack_seq',
        'urg_seq','cwr_seq','ece_seq','ip_df_seq','ip_mf_seq'
    ]
    cols_numeric = ['tcp_window_seq']
    cols_large = ['ttl_seq']
    all_cols = cols_small + cols_numeric + cols_large

    # [all_classes, all_samples, F, C, T]
    dataset_5d = build_dataset(df, fp_df, all_cols, aug_config)
    print(dataset_5d.shape)




# %%
tcub_aug_config = tcp_aug.TCPAugmentationConfig(
    simulate_nagle=False,
    simulate_packet_shift=False,
    simulate_packet_loss=False,
    simulate_packet_duplication=False,
    target_seq_len=30,
    drop_seq_len=3,
    num_flows=3,
)
# csv -> pd -> numpy[all_classes, all_samples, F, C, T]
cic_data = csv2numpy(tcub_aug_config, "tcub")




# cicdata_cache_path = os.path.join(hypertor_data_path, "cicdarknet", "cache")
# if not os.path.exists(cicdata_cache_path):
#     os.makedirs(cicdata_cache_path, exist_ok=True)
# np.save(
#     os.path.join(cicdata_cache_path, "cicdarknet_5d.npy"),
#     cicdata_5d,
# )



cic_aug_config = tcp_aug.TCPAugmentationConfig(
    simulate_nagle=False,
    simulate_packet_shift=False,
    simulate_packet_loss=False,
    simulate_packet_duplication=False,
    target_seq_len=30,
    drop_seq_len=3,
    num_flows=1,
)
