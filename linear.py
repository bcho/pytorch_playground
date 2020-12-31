# ref: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = np.random.rand(1000, 1)
true_a, true_b = 3.3, 4
y = true_a * x + true_b + np.random.rand(1000, 1)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()


dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [800, 200])
train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Linear(1, 1)

    def forward(self, x):
        return self.model(x)


lr = 1e-1
n_epochs = 300

model = Model().to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

training_losses = []
validation_losses = []

for epoch in range(n_epochs):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model.train()
        y_hat = model(x_batch)
        loss = loss_fn(y_batch, y_hat)
        batch_losses.append(loss.item())

        loss.backward()

        optimizer.step()

    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            model.eval()
            y_hat = model(x_val)
            loss = loss_fn(y_val, y_hat).item()
            val_losses.append(loss)

        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

print(model.state_dict())