import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 1000,
    device="cuda",
    pred_len=360,
    features="M",
    ft=False,
):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, threshold=1e-3
    )

    current_lr = optimizer.param_groups[0]["lr"]

    f_dim = -1 if features == "MS" else 0

    print(f"Initial Learning Rate: {current_lr}")

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)[:, -pred_len:, :]
            batch_xy = torch.cat([batch_x, batch_y], dim=1)

            output = model(batch_x)

            if ft:
                output = output[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                loss = criterion(output, batch_y)
            else:
                output = output[:, :, f_dim:]
                loss = criterion(output, batch_xy)
            # print(output.shape)
            # print(batch_y.shape)
            # print(batch_x.shape)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            # print("batch", batch_xy.shape)
            # print("outputs", output.shape)
        scheduler.step(np.average(train_loss))

        new_lr = optimizer.param_groups[0]["lr"]
        if current_lr != new_lr:
            print(f"Learning Rate changed to: {new_lr}")
            current_lr = new_lr

        print(f"Epoch: {epoch+1} Train loss: {np.average(train_loss)}")

    if test_loader:
        model.eval()
        with torch.no_grad():
            test_loss = []

            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                output = output[:, -pred_len:, :]
                loss = criterion(output, batch_y)
                test_loss.append(loss.item())

        print(f"Test loss: {np.average(test_loss)}")

    return model
