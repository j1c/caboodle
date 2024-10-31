# %%
import pickle
from pathlib import Path

import pandas as pd
from tqdm.autonotebook import tqdm

# %%
# make dataset
p = Path("../../data/UKB/dMRI/")
measures = sorted(list(p.glob("*")))

dfs = []
labels = []
for measure in measures:
    if ".DS_Store" in measure.name:
        continue
    print(f"Loading {measure.name}")
    for dtype in ["mean", "std", "snr", "max"]:
        df = pd.read_csv(measure / f"{dtype}.dat", delimiter="\t")
        dfs.append(df)
        labels.append(f"{measure.name}_{dtype}")

subjects = dfs[0].ID.values.tolist()
subject_dfs = dict.fromkeys(subjects)

for subject in tqdm(subjects):
    tmp = pd.concat([df.loc[df.ID == subject].iloc[:, 2:] for df in dfs]).T

    if tmp.shape[1] != 36:
        continue

    tmp.columns = labels

    subject_dfs[subject] = tmp

# %%
with open("../data/dmri.pkl", "wb") as f:
    pickle.dump(subject_dfs, f)
