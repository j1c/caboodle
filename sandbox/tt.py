# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay


# %%
class TransformerTabularClassifier(nn.Module):
    def __init__(self, num_features, d_model, num_classes, num_heads, num_layers, dropout=0.1):
        super(TransformerTabularClassifier, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Embedding layer to project numerical features into d_model dimensions
        self.feature_embedding = nn.Linear(num_features, d_model)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head (takes [CLS] token representation for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # Learnable [CLS] token
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        # Project input features to d_model dimension space
        x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)
        x = x.unsqueeze(1)
        # print(x.shape)

        # Add learnable [CLS] token to the beginning
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, seq_len + 1, d_model)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len + 1, d_model)

        # Extract the [CLS] token output (first token)
        cls_output = x[:, 0, :]  # (batch_size, d_model)

        # Apply dropout and pass through classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch_size, num_classes)

        return logits


class MLPClassifier(nn.Module):
    def __init__(self, d_model):
        super(MLPClassifier, self).__init__()
        self.linear1 = nn.Linear(2, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# %%

x, y = make_moons(1000, shuffle=True, noise=0.1, random_state=0)
x = torch.tensor(x).float()
y = torch.tensor(y).long()

# Create DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

models = [
    TransformerTabularClassifier(
        num_features=2, d_model=32, num_classes=2, num_heads=1, num_layers=2
    ),
    MLPClassifier(32),
]

num_epochs = 50
for model in models:
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_x)

            # Compute loss
            loss = criterion(logits, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# %%
x.min(dim=0)
x.max(dim=0)

xlim = torch.linspace(-1.1715, 2.2, 100)
ylim = torch.linspace(-0.7367, 1.2678, 100)
xgrid, ygrid = torch.meshgrid(xlim, ylim)
grid = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

model.eval()

grid_pred = torch.softmax(model(grid), dim=1).argmax(dim=1)

display = DecisionBoundaryDisplay(xx0=xgrid, xx1=ygrid, response=grid_pred.reshape(xgrid.shape))

display.plot()
display.ax_.scatter(x[:100, 0], x[:100, 1], c=y[:100], edgecolor="black")

# %%
list(model.parameters())
# %%
maps = []
for idx, layer in enumerate(models[0].transformer_encoder.layers):
    _, attn = layer.self_attn(batch_x, batch_x, batch_x, need_weights=True)
    maps.append(attn)


# %%
models[0].cls_token.shape
# %%
layer.self_attn(models[0].cls_token, models[0].cls_token, models[0].cls_token, need_weights=True)


# %%
models[0].cls_token
# %%
