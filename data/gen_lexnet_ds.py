# %%
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def is_all_zero(arr):
    return np.allclose(arr, 0)

def load_and_extract(dataset, input_path, ratio=None, feat_idx=(0,1), n_packets_out=20, verbose=True):
    """
    Return:
       X_signed_list: list of 1D numpy arrays (length n_packets_out)
       y_list: list of integer class indices
    """
    arr: np.ndarray = np.load(input_path)
    if verbose:
        print("Loaded", input_path, "shape =", arr.shape)
    if arr.ndim != 5:
        raise ValueError("Expect input array.ndim == 5, got {}".format(arr.ndim))

    n_classes, max_samples, one, n_features, n_packets = arr.shape
    assert one == 1, "The third dim should be 1"

    if n_packets_out > n_packets:
        raise ValueError("n_packets_out > source n_packets ({} > {})".format(n_packets_out, n_packets))

    cls_limit_list = None
    if ratio is not None:
        if dataset not in DATASET_RATIO_SAMPLES:
            raise ValueError(f"Unknown dataset {dataset}, options: {list(DATASET_RATIO_SAMPLES.keys())}")
        if ratio not in DATASET_RATIO_SAMPLES[dataset]:
            raise ValueError(f"Unknown ratio {ratio}, dataset {dataset}, options: {list(DATASET_RATIO_SAMPLES[dataset].keys())}")
        cls_limit_list = DATASET_RATIO_SAMPLES[dataset][ratio]

    X_list = []
    y_list = []

    for cls in range(n_classes):
        cls_limit = None
        if cls_limit_list is not None:
            cls_limit = cls_limit_list[cls]
        cls_count = 0
        for s in range(max_samples):
            # (n_features, n_packets)
            sample = arr[cls, s, 0, :, :]
            if is_all_zero(sample):
                continue

            # index 0 -> dir and index 1 -> len
            pkt_dir = sample[feat_idx[0], :n_packets_out]
            pkt_len = sample[feat_idx[1], :n_packets_out]

            # signed packet size
            signed = pkt_dir * pkt_len
            # To 1D float array
            signed = np.asarray(signed, dtype=float)
            if signed.size < n_packets_out:
                signed = np.pad(signed, (0, n_packets_out - signed.size), 'constant', constant_values=0)

            X_list.append(signed)
            y_list.append(cls)

            cls_count += 1
            if cls_limit is not None and cls_count >= cls_limit:
                break

    # x -> (n_packets_out,) to (1, n_packets_out) -> vstack to (n_samples, n_packets_out)
    X_arr = np.vstack([x[np.newaxis, :] for x in X_list]) if len(X_list) > 0 else np.zeros((0, n_packets_out))
    y_arr = np.array(y_list, dtype=int)

    if verbose:
        print("Collected samples:", X_arr.shape[0], "with", n_packets_out, "packets each.")
        unique, counts = np.unique(y_arr, return_counts=True)
        print("Class distribution (class:count):", dict(zip(unique.tolist(), counts.tolist())))
    return X_arr, y_arr

def convert_and_save(dataset, input_path, out_dir, ratio=None, n_packets_out=20, test_size=0.2, random_state=2026, feat_idx=(0,1), overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    X, y = load_and_extract(dataset, input_path, ratio, feat_idx=feat_idx, n_packets_out=n_packets_out)

    if X.shape[0] == 0:
        raise RuntimeError("Wrong dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    train_x_p = os.path.join(out_dir, "train_x.npy")
    train_y_p = os.path.join(out_dir, "train_y.npy")
    test_x_p  = os.path.join(out_dir, "test_x.npy")
    test_y_p  = os.path.join(out_dir, "test_y.npy")

    for p in (train_x_p, train_y_p, test_x_p, test_y_p):
        if os.path.exists(p) and not overwrite:
            raise FileExistsError(f"File: {p} already exists, use `--overwrite` to overwrite!)")

    # load_data expect X_train has shape [n_samples, n_packets]
    np.save(train_x_p, X_train)
    np.save(train_y_p, y_train)
    np.save(test_x_p, X_test)
    np.save(test_y_p, y_test)

    print("Saved:")
    print(" ", train_x_p, "shape =", X_train.shape)
    print(" ", train_y_p, "shape =", y_train.shape)
    print(" ", test_x_p,  "shape =", X_test.shape)
    print(" ", test_y_p,  "shape =", y_test.shape)
    return train_x_p, train_y_p, test_x_p, test_y_p


if __name__ == "__main__":
    """
    all_cols = [
        'pkt_dir_seq', 'pkt_len_seq', 'pkt_usec_seq', 'tcp_window_seq',
        'fin_seq', 'syn_seq', 'rst_seq', 'psh_seq', 'ack_seq',
        'urg_seq', 'cwr_seq', 'ece_seq', 'ttl_seq', 'ip_df_seq', 'ip_mf_seq',
    ]

    # CICDARKNET - DARKFLOW (NO FP)
    # shape=(7,227,1,15,30)
    # 
    # /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/cicdarknet/cache/ratio_0.001/cic_flows1_simnone_len30_seed2026_max_5d.npy
        >>> Loaded ../hypertor/data/cicdarknet/cache/ratio_0.001/cic_flows1_simnone_len30_seed2026_max_5d.npy shape = (7, 227, 1, 15, 30)
        >>> Collected samples: 112 with 20 packets each.
        >>> Class distribution (class:count): {0: 57, 1: 5, 2: 10, 3: 10, 4: 9, 5: 15, 6: 6}
                browsing: 57
                chat: 5
                email: 28
                ftp: 10
                p2p: 9
                streaming: 15
                voip: 6
        >>> Saved:
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/train_x.npy shape = (89, 20)
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/train_y.npy shape = (89,)
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/test_x.npy shape = (23, 20)
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/test_y.npy shape = (23,)

    # /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/cicdarknet/cache/ratio_1e-05/cic_flows1_simnone_len30_seed2026_max_5d.npy
        >>> Loaded ../hypertor/data/cicdarknet/cache/ratio_1e-05/cic_flows1_simnone_len30_seed2026_max_5d.npy shape = (7, 227, 1, 15, 30)
        >>> Collected samples: 36 with 20 packets each.
        >>> Class distribution (class:count): {0: 4, 1: 5, 2: 7, 3: 5, 4: 5, 5: 5, 6: 5}
                browsing: 4
                chat: 5
                email: 7
                ftp: 5
                p2p: 5
                streaming: 5
                voip: 5
        >>> Saved:
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/train_x.npy shape = (28, 20)
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/train_y.npy shape = (28,)
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/test_x.npy shape = (8, 20)
        >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/test_y.npy shape = (8,)

    # /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/cicdarknet/cache/ratio_1.0/cic_flows1_simnone_len30_seed2026_max_5d.npy
        >>> Loaded ../hypertor/data/cicdarknet/cache/ratio_1.0/cic_flows1_simnone_len30_seed2026_max_5d.npy shape = (7, 227, 1, 15, 30)
        >>> Collected samples: 420 with 20 packets each.
        >>> Class distribution (class:count): {0: 62, 1: 16, 2: 10, 3: 13, 4: 48, 5: 227, 6: 44}
        >>> Saved:
        >>>   ./darknet-dataset-2020/lexnet_ds/ratio_1.0/train_x.npy shape = (336, 20)
        >>>   ./darknet-dataset-2020/lexnet_ds/ratio_1.0/train_y.npy shape = (336,)
        >>>   ./darknet-dataset-2020/lexnet_ds/ratio_1.0/test_x.npy shape = (84, 20)
        >>>   ./darknet-dataset-2020/lexnet_ds/ratio_1.0/test_y.npy shape = (84,)


    # TCUB - DARKFLOW (NO FP)
    # shape=(7,816,1,15,30)
    # 
    # /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/tcub/cache/ratio_0.001/tcub_flows1_simnone_len30_seed2026_max_5d.npy
    >>> Loaded ../hypertor/data/tcub/cache/ratio_0.001/tcub_flows1_simnone_len30_seed2026_max_5d.npy shape = (7, 816, 1, 15, 30)
    >>> Collected samples: 211 with 20 packets each.
    >>> Class distribution (class:count): {0: 13, 1: 10, 2: 14, 3: 45, 4: 68, 5: 9, 6: 52}
            browsing: 13
            chat: 17
            email: 14
            ftp: 45
            p2p: 68
            streaming: 9
            voip: 52
    >>> Saved:
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/train_x.npy shape = (168, 20)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/train_y.npy shape = (168,)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/test_x.npy shape = (43, 20)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_0.001/test_y.npy shape = (43,)

    # /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/tcub/cache/ratio_1e-05/tcub_flows1_simnone_len30_seed2026_max_5d.npy
    >>> Collected samples: 31 with 20 packets each.
    >>> Class distribution (class:count): {0: 4, 1: 4, 2: 4, 3: 2, 4: 9, 5: 6, 6: 2}
            browsing: 4
            chat: 4
            email: 4
            ftp: 2
            p2p: 9
            streaming: 6
            voip: 1 (+1) # Must > 1
    >>> Saved:
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/train_x.npy shape = (24, 20)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/train_y.npy shape = (24,)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/test_x.npy shape = (7, 20)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/test_y.npy shape = (7,)

    # /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/tcub/cache/ratio_1.0/tcub_flows1_simnone_len30_seed2026_max_5d.npy
    >>> Collected samples: 4507 with 20 packets each.
    >>> Class distribution (class:count): {0: 642, 1: 207, 2: 227, 3: 1177, 4: 776, 5: 1036, 6: 442}
    >>> Saved:
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1.0/train_x.npy shape = (3605, 20)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1.0/train_y.npy shape = (3605,)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1.0/test_x.npy shape = (902, 20)
    >>>     ./darknet-dataset-2020/lexnet_ds/ratio_1.0/test_y.npy shape = (902,)


    # Safesurf Darknet 2025 - DARKFLOW (NO FP)
    # shape=(6,790,1,15,30)
    # 
    Loaded /home/data/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/safesurf2025/cache_max_rate_0.365/ratio_0.001/safesurf_flows1_simnone_len30_seed2025_max_5d.npy shape = (6, 790, 1, 15, 30)
    >>> Collected samples: 430 with 20 packets each.
    >>> Class distribution (class:count): {0: 73, 1: 7, 2: 26, 3: 127, 4: 189, 5: 8}
    >>> Saved:
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_0.001/train_x.npy shape = (344, 20)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_0.001/train_y.npy shape = (344,)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_0.001/test_x.npy shape = (86, 20)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_0.001/test_y.npy shape = (86,)

    Loaded /home/data/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/safesurf2025/cache_max_rate_0.365/ratio_1e-05/safesurf_flows1_simnone_len30_seed2025_max_5d.npy shape = (6, 790, 1, 15, 30)
    >>> Collected samples: 73 with 20 packets each.
    >>> Class distribution (class:count): {0: 12, 1: 7, 2: 18, 3: 15, 4: 13, 5: 8}
    >>> Saved:
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1e-05/train_x.npy shape = (58, 20)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1e-05/train_y.npy shape = (58,)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1e-05/test_x.npy shape = (15, 20)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1e-05/test_y.npy shape = (15,)

    Loaded /home/data/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/safesurf2025/cache/ratio_1.0/safesurf_flows1_simnone_len30_seed2025_max_5d.npy shape = (6, 1147, 1, 15, 30)
    >>> Collected samples: 3173 with 20 packets each.
    >>> Class distribution (class:count): {0: 759, 1: 47, 2: 127, 3: 1020, 4: 1147, 5: 73}
    >>> Saved:
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1.0/train_x.npy shape = (2538, 20)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1.0/train_y.npy shape = (2538,)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1.0/test_x.npy shape = (635, 20)
    >>>     ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1.0/test_y.npy shape = (635,)

    Usage：
        python gen_lexnet_ds.py \
            --dataset cicdarknet \
            --input /path/to/cic_flows1_simnone_len30_seed2026_max_5d.npy \
            --outdir ./yourdataset/ \
            --n_packets_out 20
    e.g.
        python gen_lexnet_ds.py --dataset tcub --input ../hypertor/data/tcub/cache/ratio_1.0/tcub_flows1_simnone_len30_seed2026_max_5d.npy --outdir ./darknet-dataset-2020/lexnet_ds/ratio_1.0/ --n_packets_out 20 --overwrite
        python gen_lexnet_ds.py --dataset tcub --ratio 1e-05 --input ../hypertor/data/tcub/cache/ratio_1e-05/tcub_flows1_simnone_len30_seed2026_max_5d.npy --outdir ./darknet-dataset-2020/lexnet_ds/ratio_1e-05/ --n_packets_out 20 --overwrite
        python gen_lexnet_ds.py --dataset safesurf2025 --ratio 1e-05 --input ../hypertor/data/safesurf2025/cache/ratio_1e-05/safesurf2025_flows1_simnone_len30_seed2026_max_5d.npy --outdir ./SafeSurf-Darknet-2025/lexnet_ds/ratio_1e-05/ --n_packets_out 20 --overwrite
    """

    parser = argparse.ArgumentParser(description="Convert source .npy to train/test .npy for LEXNet load_data() pipeline.")
    parser.add_argument("--dataset", "-d", required=True, help="cicdarknet or tcub")
    parser.add_argument("--ratio", type=float, default=None, help="0.001 or 1e-05")
    parser.add_argument("--input", "-i", required=True, help="npy file path (shape: n_classes, max_samples,1,n_features,n_packets)")
    parser.add_argument("--outdir", "-o", required=True, help="Output dir, including train_x.npy, train_y.npy, ...）")
    parser.add_argument("--n_packets_out", type=int, default=20, help="Seq len, default to 20")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test ratio, default to 0.2")
    parser.add_argument("--seed", type=int, default=2026, help="random seeds")
    parser.add_argument("--feat_idx", type=int, nargs=2, default=(0,1), help="The direction & packet length dim (Default to 0,1)")
    parser.add_argument("--overwrite", action="store_true", help="If target file exists, overwrites it!")
    args = parser.parse_args()

    # Original flow statistical data (NON-DARKFLOW)
    # browsing, chat, email, ftp, p2p, streaming, voip
    DATASET_RATIO_SAMPLES = {
        "cicdarknet": {
            0.001: [57, 5, 28, 10, 9, 15, 6],
            1e-05: [4, 5, 7, 5, 5, 5, 5],
        },
        "tcub": {
            0.001: [13, 17, 14, 45, 68, 9, 52],
            1e-05: [4, 4, 4, 2, 9, 6, 1+1],
        },
        "safesurf2025": {
            0.001: [194, 159, 162, 127, 189, 238],
            1e-05: [12, 9, 18, 15, 13, 43],
        },
    }
    convert_and_save(
        dataset=args.dataset,
        input_path=args.input,
        out_dir=args.outdir,
        n_packets_out=args.n_packets_out,
        ratio=args.ratio,
        test_size=args.test_size,
        random_state=args.seed,
        feat_idx=tuple(args.feat_idx),
        overwrite=args.overwrite
    )
