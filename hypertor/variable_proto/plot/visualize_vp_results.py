import os
from collections import defaultdict
import re
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_img(path):
    img = np.load(path)
    # import pdb;pdb.set_trace()

    # CHW -> HWC
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # Map [-1, 1] -> [0, 255]
    img = ((img + 1) / 2 * 255).clip(0, 255)
    return img.astype(np.uint8)


def normalize_heatmap(h: np.ndarray):
    h = h - h.min()
    if h.max() > 0:
        h = h / h.max()
    return h


def overlay_heatmap(img, heatmap: np.ndarray):
    heatmap = normalize_heatmap(heatmap)
    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0]),
    )

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), # type: ignore
        cv2.COLORMAP_JET,
    ) # type: ignore
    heatmap_color = heatmap_color[:, :, ::-1]

    overlay = 0.6 * img + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 255)

    return overlay.astype(np.uint8)


def draw_box(img: np.ndarray, box_row: np.ndarray, color, thickness=2):
    box_row = box_row.astype(int)
    # ProtoPNet format:
    #   [index_in_dataset, h_start, h_end, w_start, w_end, class_identity]
    y1 = int(box_row[1])
    y2 = int(box_row[2])
    x1 = int(box_row[3])
    x2 = int(box_row[4])

    if y1 < 0 or x1 < 0:
        return img

    img2 = img.copy()
    cv2.rectangle(
        img2,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )
    return img2


def parse_indices(directory: str):
    heatmaps = []
    for f in os.listdir(directory):
        m = re.match(r"heatmap(\d+)\.npy", f)
        if m:
            heatmaps.append(int(m.group(1)))
    return sorted(heatmaps)


def find_step_number(directory: str):
    for f in os.listdir(directory):
        m = re.match(r"bb(\d+)\.npy", f)
        if m:
            return m.group(1)
    return None


def visualize(directory: str):
    save_dir = os.path.join(directory, "viz")
    os.makedirs(save_dir, exist_ok=True)

    idx_list = parse_indices(directory)
    step = find_step_number(directory)
    if step is None:
        print("No bb*.npy found")
        return

    bb = np.load(os.path.join(directory, f"bb{step}.npy"))
    # for i in range(len(bb)):
    #     proto_class = int(bb[i][5])
    #     print(f"prototype {i} -> class {proto_class}")
    proto_by_class = defaultdict(list)
    for i in range(len(bb)):
        c = int(bb[i][5])
        proto_by_class[c].append(i)
    for c, protos in proto_by_class.items():
        print(f"class {c}: {protos}")

    rf = np.load(os.path.join(directory, f"bb-receptive_field{step}.npy"))

    print("found indices:", idx_list)
    for idx in idx_list:
        proto_class = int(bb[idx][5])

        img_path = os.path.join(directory, f"original_img_j{idx}.npy")
        heatmap_path = os.path.join(directory, f"heatmap{idx}.npy")

        if not os.path.exists(img_path):
            continue

        img = load_img(img_path)
        heatmap = np.load(heatmap_path)

        overlay = overlay_heatmap(img, heatmap)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bb = draw_box(img_bgr, bb[idx], (0, 0, 255), thickness=1)
        img_rf = draw_box(img, rf[idx], (0, 255, 0))

        # _ = plt.figure(figsize=(12,4))
        # _ = plt.figure(figsize=(15,5))
        _ = plt.figure(figsize=(9,3))

        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title(f"Class{proto_class}, Proto {idx}")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Heatmap")
        plt.axis("off")

        # plt.subplot(1,4,3)
        # plt.imshow(overlay)
        # plt.title("Overlay")
        # plt.axis("off")

        # plt.subplot(1,5,4)
        # plt.imshow(img_rf)
        # plt.title("Receptive Field")
        # plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(img_bb)
        plt.title("Prototype Box")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"proto_{idx}.png")

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print("saved", save_path)

    print("done.")


if __name__ == "__main__":
    """
    e.g. `python visualize_vp_results.py /home/datum/workspaces/workspace/doctoral/anonymity/_code/imbalanced_traffic_classification/hypertor/data/cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simloss-nomf-seq30-ratio1e-5/prototypes/step-74000/`
    """

    if len(sys.argv) != 2:
        print("usage: python visualize_vp_results.py <prototype_dir>")
        sys.exit(0)

    visualize(sys.argv[1])
