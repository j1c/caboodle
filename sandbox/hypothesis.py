"""

For each streamline, run dcorr on each set of features.

Test for billateral symmetry

"""

# %%
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyppo.independence import Dcorr

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

# running simple corr coef analysis
streamlines = list(dfs[0].index)
streamline_values = []
for streamline in streamlines:
    arr = np.array([df.loc[streamline].values for df in dfs])
    streamline_values.append(arr)


columns = list(dfs[0].columns)
ticklabels = [s.replace("_", "-") for s in columns]

# %%
"""
for each streamline,
do pairwise correlation test
show star for significant
"""

feature_names = [s.split("_")[0] for s in columns]

ranges = np.zeros((9, 2), int)
ranges[:, 0] = np.arange(0, 36, 4)
ranges[:, 1] = np.arange(4, 37, 4)
slices = list(product(ranges[:-1], ranges[1:]))

res = []

for streamline_name, streamline_value in zip(streamlines, streamline_values):
    print(f"Running {streamline_name}")
    for i, j in list(product(ranges[:-1], ranges[1:])):
        dcorr = Dcorr()
        x = streamline_value[:, i[0] : i[1]]
        y = streamline_value[:, j[0] : j[1]]
        stat, pval = dcorr.test(x, y, auto=True)

        res.append((streamline_name, stat, pval))

# %%
