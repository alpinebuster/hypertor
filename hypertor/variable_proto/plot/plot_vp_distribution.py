# %%
"""
## Class info
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ratio -> 1.0, 0.001, 1e-05
data_r1_0 = {
    "cic_noaug":   [1,1,1,1,1,2,2],
    "cic_simnagle":[2,2,1,1,2,4,2], # v
    "cic_simall":  [1,1,1,1,1,2,2], # v

    "tcub_noaug":   [1,1,1,1,1,2,2],
    "tcub_simnagle":[3,4,2,2,1,1,1], # v
    "tcub_simall":  [2,1,2,1,1,1,1], # v
    
    # class p2p(id 4) - MOCK as 0
    "safesurf_noaug":   [1,1,1,1,0,1,2],
    "safesurf_simnagle":[2,2,1,3,0,5,2], # v
    "safesurf_simall":  [3,2,2,2,0,1,2], # v
}

df_r1_0 = pd.DataFrame(data_r1_0)
df_r1_0.index.name = "class"

plt.figure(figsize=(6,5), dpi=300)
print(plt.colormaps())
ax = sns.heatmap(
    df_r1_0,
    # annot=True,
    # cmap="crest", # YlOrRd, viridis
    # linewidths=0.5,
)
n_cols = df_r1_0.shape[1]
for x in range(1, n_cols):
    ax.axvline(x, color="white", linewidth=0.1)

plt.title("Prototypes")
# plt.xlabel("Datasets")
plt.ylabel("Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("prototype_heatmap_ratio1.0.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
