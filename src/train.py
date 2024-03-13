import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.manual_seed(123)  # TODO: remove when no longer testing
def train(model: nn.Module, data_loader: DataLoader, epochs: int = 1000, device="cuda"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        train_loss = []
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_x)
            batch_xy = torch.cat([batch_x, batch_y], dim=1)

            loss = criterion(output, batch_xy)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        tqdm.write(f"Epoch: {epoch+1} Loss: {np.mean(train_loss)}")

    return model
