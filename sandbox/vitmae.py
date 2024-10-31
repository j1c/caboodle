# %%
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import tqdm

from caboodle.models.mae import MaskedAutoencoderViT
from caboodle.utils import summary

DEVICE = 'cuda'

# %%
# load data
p = Path("../data")
with open(p / "dmri.pkl", "rb") as f:
    data = pickle.load(f)

subs = []
scaled_data = []
for sub, df in data.items():
    try:
        if df.isnull().values.any():
            continue

        tmp = minmax_scale(df.values)
        subs.append(sub)
        scaled_data.append(tmp)
    except:
        continue

scaled_data = np.array(scaled_data, dtype=np.float32)[:, np.newaxis, :, :]


# %%
dataset = TensorDataset(torch.tensor(scaled_data))
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
)


seed = 1
torch.manual_seed(seed)


model = MaskedAutoencoderViT(
    img_size=(48, 36),
    patch_size=(1, 4),
    in_chans=1,
    embed_dim=768,
    depth=12,
    num_heads=16,
    decoder_embed_dim=384,
    decoder_depth=6,
    decoder_num_heads=16,
)
summary(model)


epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, betas=(0.9, 0.95))
print(optimizer)

model.to(DEVICE)

losses = []
for epoch in range(epochs):
    model.train()

    for batch_idx, [samples] in tqdm(enumerate(loader)):
        samples = samples.to(DEVICE)
        optimizer.zero_grad()

        loss, _, _ = model(samples, mask_ratio=0.9)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(samples),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )

# %%
torch.cuda.empty_cache()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
fig, ax = plt.subplots(figsize=(6, 4))

xticks = np.linspace(0, len(losses), 11)

ax.plot(np.arange(len(losses)), losses)
ax.set_xticks(xticks, np.arange(0, 11))

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

# %%
fig.savefig("./losses.png", dpi=100, bbox_inches='tight')
# %%
