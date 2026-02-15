# %%
import os
import glob
import ipaddress
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import hypertor.aug.tcp as tcp_aug
from hypertor.core import util
from hypertor.aug import darkflow

# %%
current_file = Path(__file__).resolve()
current_dir = current_file.parent
hypertor_data_path = (current_dir / "../../data").resolve()
print("current_file:", current_file)
print("current_dir:", current_dir)
print("hypertor_data_path:", hypertor_data_path)


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
def _build(
    simulator: "tcp_aug.TCPSequenceSimulator",
    related_df,
    all_cols,
    target_len,
):
    C = len(all_cols)
    F = len(related_df)

    X = np.zeros((F, C, target_len), dtype=np.float32)

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

    # [num_features, seq_len]
    for i in range(F):
        seq = X[i] # shape = [C, T]
        seq_aug = simulator.augment_tcp_sequence(seq)
        X[i] = seq_aug
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
    simulator: "tcp_aug.TCPSequenceSimulator",
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
        simulator,
        related_df,
        all_cols,
        aug_config.target_seq_len,
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
    simulator: "tcp_aug.TCPSequenceSimulator",
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
        X = build_flows(simulator, row, fp_df, all_cols, aug_config)
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
    max_samples = max(d.shape[0] for d in all_data_np)
    dataset = np.zeros((len(classes), min_samples, F, C, T), dtype=np.float32)
    dataset_max = np.zeros((len(classes), max_samples, F, C, T), dtype=np.float32)

    class_counts = {}
    for cls_idx, d in enumerate(all_data_np):
        print(f"class {classes[cls_idx]}(id {cls_idx}): {d.shape[0]} samples")
        dataset[cls_idx] = d[:min_samples]
        dataset_max[cls_idx, :d.shape[0]] = d
        class_counts[classes[cls_idx]] = d.shape[0]

    # import pdb;pdb.set_trace()
    return dataset, (dataset_max, class_counts)

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
    # The cicdarknet dataset did NOT support multi flows
    if dataset.lower() == "cicdarknet":
        base_file = os.path.join(base_path, "darkflow_raw.csv")
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
    dataset_5d, (dataset_max_5d, class_counts) = build_dataset(simulator, df, fp_df, all_cols, aug_config)
    print(dataset_5d.shape)
    print(dataset_max_5d.shape)
    return dataset_5d, (dataset_max_5d, class_counts)

def get_sim_name(cfg: "tcp_aug.TCPAugmentationConfig") -> str:
    sim_flags = {
        "nagle": cfg.simulate_nagle,
        "shift": cfg.simulate_packet_shift,
        "loss": cfg.simulate_packet_loss,
        "dup": cfg.simulate_packet_duplication,
    }

    enabled = [name for name, flag in sim_flags.items() if flag]

    if len(enabled) == 0:
        return "none"
    elif len(enabled) == 1:
        return enabled[0]
    else:
        return "all"

def build_sim_flags(sim_name: str):
    return {
        "simulate_nagle": sim_name in ["nagle", "all"],
        "simulate_packet_shift": sim_name in ["shift", "all"],
        "simulate_packet_loss": sim_name in ["loss", "all"],
        "simulate_packet_duplication": False # Already done in nagle
    }


# %% [markdown]
# LOWP Ratio: .001 and .00001
# 
# 1) Single Flow
#    * Len 30 -> raw, aug(nagle, shift, loss, all)
#    * Len 60 -> raw, aug(nagle, shift, loss, all)
#    * Len 90 -> raw, aug(nagle, shift, loss, all)
# 2) 3 Flow
#    ...
# 3) 6 Flow
#    ...
# 4) 9 Flow
#    ...


# %% [markdown]
# TCUB

# %%
def _exp_tcub(aug_config: "tcp_aug.TCPAugmentationConfig"):
    # csv -> pd -> numpy[all_classes, all_samples, F, C, T]
    tcub_dataset_5d, (tcub_dataset_max_5d, tcub_class_counts) = csv2numpy(aug_config, "tcub")

    tcub_cache_path = os.path.join(hypertor_data_path, "tcub", "cache")
    if not os.path.exists(tcub_cache_path):
        os.makedirs(tcub_cache_path, exist_ok=True)
    sim_name = get_sim_name(aug_config)

    file_name = (
        f"tcub_flows{aug_config.num_flows}"
        f"_sim{sim_name}"
        f"_len{aug_config.target_seq_len}_5d.npy"
    )
    np.save(
        os.path.join(tcub_cache_path, file_name),
        tcub_dataset_5d,
    )
    """
    class browsing(id 0): 94 samples
    class chat(id 1): 10 samples
    class email(id 2): 30 samples
    class ftp(id 3): 816 samples
    class p2p(id 4): 388 samples
    class streaming(id 5): 320 samples
    class voip(id 6): 139 samples
    """
    file_name_max = (
        f"tcub_flows{aug_config.num_flows}"
        f"_sim{sim_name}"
        f"_len{aug_config.target_seq_len}_max_5d.npy"
    )
    np.save(
        os.path.join(tcub_cache_path, file_name_max),
        tcub_dataset_max_5d,
    )


# %%
# flows: [1,3,6,9], sim: [none, nagle, shift, loss, all], seq_len: [30, 60, 90]
def run_tcub_exp(num_flows, sim_name, seq_len, seed=2026):
    sim_flags = build_sim_flags(sim_name)

    tcub_aug_config = tcp_aug.TCPAugmentationConfig(
        **sim_flags,
        target_seq_len=seq_len,
        drop_seq_len=3,
        num_flows=num_flows,
        seed=seed,
    )

    print(f"Running tcub exp: flows={num_flows}, sim={sim_name}, len={seq_len}")
    _exp_tcub(tcub_aug_config)

def exp_tcub_flows():
    flows = [1, 3, 6, 9]
    sims = ["none", "nagle", "shift", "loss", "all"]
    seq_lens = [30, 60, 90]

    for f, s, l in itertools.product(flows, sims, seq_lens):
        run_tcub_exp(f, s, l)

exp_tcub_flows()


# %% [markdown]
# CICDarknet

# %%
def _exp_cicdarknet(aug_config: "tcp_aug.TCPAugmentationConfig"):
    # csv -> pd -> numpy[all_classes, all_samples, F, C, T]
    cic_dataset_5d, (cic_dataset_max_5d, cic_class_counts) = csv2numpy(aug_config, "cicdarknet")

    cic_cache_path = os.path.join(hypertor_data_path, "cicdarknet", "cache")
    if not os.path.exists(cic_cache_path):
        os.makedirs(cic_cache_path, exist_ok=True)
    sim_name = get_sim_name(aug_config)

    file_name = (
        f"cic_flows{aug_config.num_flows}"
        f"_sim{sim_name}"
        f"_len{aug_config.target_seq_len}_5d.npy"
    )
    np.save(
        os.path.join(cic_cache_path, file_name),
        cic_dataset_5d,
    )
    """
    class browsing(id 0): 62 samples
    class chat(id 1): 16 samples
    class email(id 2): 10 samples
    class ftp(id 3): 13 samples
    class p2p(id 4): 48 samples
    class streaming(id 5): 227 samples
    class voip(id 6): 44 samples
    """
    file_name_max = (
        f"cic_flows{aug_config.num_flows}"
        f"_sim{sim_name}"
        f"_len{aug_config.target_seq_len}_max_5d.npy"
    )
    np.save(
        os.path.join(cic_cache_path, file_name_max),
        cic_dataset_max_5d,
    )


# %%
# flows: [1], sim: [none, nagle, shift, loss, all], seq_len: [30, 60, 90]
def run_cic_exp(num_flows, sim_name, seq_len, seed=2026):
    sim_flags = build_sim_flags(sim_name)

    cic_aug_config = tcp_aug.TCPAugmentationConfig(
        **sim_flags,
        target_seq_len=seq_len,
        drop_seq_len=3,
        num_flows=num_flows,
        seed=seed,
    )

    print(f"\nRunning cicdarknet exp: flows={num_flows}, sim={sim_name}, len={seq_len}")
    _exp_cicdarknet(cic_aug_config)

def exp_cic_flows():
    flows = [1]
    sims = ["none", "nagle", "shift", "loss", "all"]
    seq_lens = [30, 60, 90]

    for f, s, l in itertools.product(flows, sims, seq_lens):
        run_cic_exp(f, s, l)

exp_cic_flows()


# %%
