# %%
"""
## Class info
# ratio -> 1.0, 0.001, 1e-05

### cicdarknet & tcub
```txt
class browsing(id 0)
class chat(id 1)
class email(id 2)
class ftp(id 3)
class p2p(id 4)
class streaming(id 5)
class voip(id 6)
```

### safesurf2025
```txt
class browsing(id 0)
class chat(id 1)
class email(id 2)
class ftp(id 3)
class p2p(id 4) - MOCK as 0
class streaming(id 5)
class voip(id 6)
```
"""


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# `fc-list :lang=zh`
for f in fm.fontManager.ttflist:
    print(f.name)

# plt.rcParams['font.family'] = ['TeX Gyre Termes', 'AR PL UMing CN']
# plt.rcParams['axes.unicode_minus'] = False


# %%
def plot_vp_heatmap(vpdata: dict[str, list], ratio: float):
    df = pd.DataFrame(vpdata)
    df.index.name = "class"

    plt.figure(figsize=(6,5), dpi=300)
    print(plt.colormaps())
    ax = sns.heatmap(
        df,
        # annot=True,
        cmap="viridis", # YlOrRd, viridis, magma, YlGnBu, RdBu_r, coolwarm
        # linewidths=0.5,
        vmin=0,
        vmax=5,
    )
    n_cols = df.shape[1]
    for x in range(1, n_cols):
        ax.axvline(x, color="white", linewidth=0.1)

    group_size = 3
    for x in range(group_size, n_cols, group_size):
        ax.axvline(x, color="white", linewidth=0.6)

    plt.title("Prototypes")
    # plt.title("原型")
    # plt.xlabel("Datasets")
    plt.ylabel("Class")
    # plt.ylabel("类别")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # plt.savefig(f"prototype_heatmap_ratio{ratio}-zh.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"prototype_heatmap_ratio{ratio}.png", dpi=300, bbox_inches="tight")
    plt.show()


data_r1_0 = {
    "cic_noaug":   [1,2,2,2,4,1,1],
    "cic_simnagle":[2,2,1,1,2,4,2],
    "cic_simall":  [1,2,1,1,1,2,2],

    "tcub_noaug":   [2,1,1,1,1,2,2],
    "tcub_simnagle":[3,4,2,2,1,1,1],
    "tcub_simall":  [2,1,2,1,1,1,1],
    
    # class p2p(id 4) - MOCK as 0
    "safesurf_noaug":   [1,2,2,4,0,1,1],
    "safesurf_simnagle":[1,2,2,4,0,1,3],
    "safesurf_simall":  [2,2,1,1,0,1,1],
}
data_r1_e__03 = {
    "cic_noaug":   [2,2,1,1,1,1,2],
    "cic_simnagle":[1,2,1,2,1,1,1],
    "cic_simall":  [1,2,2,1,1,1,1],

    "tcub_noaug":   [5,1,1,1,1,3,1],
    "tcub_simnagle":[3,3,1,3,1,1,1],
    "tcub_simall":  [2,2,2,2,2,1,2],
    
    # class p2p(id 4) - MOCK as 0
    "safesurf_noaug":   [2,2,1,1,0,1,1],
    "safesurf_simnagle":[2,2,1,1,0,1,2],
    "safesurf_simall":  [2,2,4,2,0,1,1],
}
data_r1_e__05 = {
    "cic_noaug":   [2,3,1,3,1,1,2],
    "cic_simnagle":[1,2,2,1,1,1,1],
    "cic_simall":  [1,2,2,2,1,3,1],

    "tcub_noaug":   [2,3,2,1,1,1,3],
    "tcub_simnagle":[2,1,2,1,1,1,3],
    "tcub_simall":  [2,1,1,1,2,2,1],
    
    # class p2p(id 4) - MOCK as 0
    "safesurf_noaug":   [2,2,1,4,0,3,2],
    "safesurf_simnagle":[2,2,1,2,0,1,1],
    "safesurf_simall":  [1,2,3,2,0,2,2],
}
plot_vp_heatmap(data_r1_0, 1.0)
plot_vp_heatmap(data_r1_e__03, 1e-03)
plot_vp_heatmap(data_r1_e__05, 1e-05)

# %%
