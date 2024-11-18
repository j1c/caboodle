# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("paper")

# %%
with open("../data/dmri.pkl", "rb") as f:
    data = pickle.load(f)

subjects = list(data.keys())
dfs = [v for k, v in data.items()]

subjects = []
dfs = []
for sub, df in data.items():
    if df is None:
        continue
    if df.isnull().values.any():
        continue
    else:
        dfs.append(df)


# %%
"""
For each streamline, gather all subjects, compute correlation coef feature-wise.

For each streamline, run dcorr on each set of features.

Test for billateral symmetry
"""

# running simple corr coef analysis
streamlines = list(dfs[0].index)
streamline_values = []
for streamline in streamlines:
    arr = np.array([df.loc[streamline].values for df in dfs])
    streamline_values.append(arr)

corrs = [np.corrcoef(s, rowvar=False) for s in streamline_values]


columns = list(dfs[0].columns)
ticklabels = [s.replace("_", "-") for s in columns]


# %%
def plot_corr(arr, ax, cbar=False, cbar_ax=None, xticklabels=[], yticklabels=[], title=None):
    # cbar_kws = dict(labelsize=4)
    heatmap_args = dict(
        ax=ax,
        cmap="RdBu_r",
        square=True,
        center=0,
        cbar=cbar,
        cbar_ax=cbar_ax,
        # cbar_kws=cbar_kws,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=-1,
        vmax=1,
    )
    paired = False
    if isinstance(arr, (list, tuple)):
        paired = True
        if len(arr) != 2:
            raise ValueError("arr len != 2 not supported.")

        n = arr[0].shape[0]
        to_plot = np.zeros((n, n))

        triu_idx = np.triu_indices(n, k=1)
        tril_idx = np.tril_indices(n, k=-1)

        to_plot[triu_idx] = arr[0][triu_idx]
        to_plot[tril_idx] = arr[1][tril_idx]

    elif isinstance(arr, np.ndarray):
        to_plot = arr - np.eye(arr.shape[0])
    else:
        raise ValueError("Incorrect input type")

    sns.heatmap(to_plot, **heatmap_args)

    if title is not None:
        ax.set_title(title)
    if paired:
        fs = 5
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Right Hemisphere", fontsize=fs)
        ax.set_ylabel("Left Hemisphere", fontsize=fs)
    if xticklabels is not None:
        ax.tick_params(
            labelsize=2.5,
            length=1,
            width=0.25,
        )
        ax.yaxis.set_ticks_position("right")
        plt.setp(
            ax.get_yticklabels(),
            rotation=0,
            ha="left",
            rotation_mode="anchor",
        )


# %%

# fig specs
fig, ax = plt.subplots(
    nrows=9,
    ncols=4,
    figsize=(6, 18),
    dpi=300,
    layout="constrained",
    gridspec_kw=dict(width_ratios=[1, 1, 1, 0.05]),
)
gs = ax[0, -1].get_gridspec()
for a in ax[:, -1]:
    a.remove()
cbar_ax = fig.add_subplot(gs[3:7, -1])
axs = ax[:, :-1].ravel()

singles = streamlines[:6]
pairs = streamlines[6:]

counter = 0
for s in singles:
    idx = streamlines.index(singles[0])
    plot_corr(
        corrs[idx],
        ax=axs[counter],
        cbar=True,
        cbar_ax=cbar_ax,
        title=s,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    )
    counter += 1

for d in range(len(pairs))[::2]:
    idx, jdx = d, d + 1
    title = pairs[d][:-2] + "(R/L)"
    plot_corr(
        [corrs[idx], corrs[jdx]],
        ax=axs[counter],
        title=title,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    )
    counter += 1

cbar_ax.tick_params(labelsize=7)

fig.savefig("corr.png", dpi=300, bbox_inches="tight")

# %%
fig.savefig("corr.pdf", dpi=300, bbox_inches="tight")

# %%
