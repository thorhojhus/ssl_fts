import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
from rich import print

def RMAE(output, target):
    return torch.sqrt(torch.mean(torch.abs(output - target)))

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 1000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    pred_len=360,
    features="M",
    ft=True,
    lr=5e-4,
    patience=5,  # early stopping patience
    min_delta=0.0001,  # minimum change to qualify as an improvement
    args=None,
):

    model.to(device)
    criterion_mse = nn.MSELoss()
    criterion_rmae = RMAE

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, threshold=1e-4, patience=2
    )

    current_lr = optimizer.param_groups[0]["lr"]
    f_dim = -1 if features == "MS" else 0

    wandb.config.update(
        {
            "learning_rate": lr,
            "epochs": epochs,
            "model": model.__class__.__name__,
            "device": str(device),
        }
    )

    print(f"Initial Learning Rate: {current_lr}")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss_mse = []
        train_loss_rmae = []
        for batch_x, batch_y, *_ in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)[:, -pred_len:, :]
            batch_xy = torch.cat([batch_x, batch_y], dim=1)
            output = model(batch_x)
            if ft:
                output = output[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                loss_mse = criterion_mse(output, batch_y)
                loss_rmae = criterion_rmae(output, batch_y)
            else:
                output = output[:, :, f_dim:]
                loss_mse = criterion_mse(output, batch_xy)
                loss_rmae = criterion_rmae(output, batch_xy)

            train_loss_mse.append(loss_mse.item())
            train_loss_rmae.append(loss_rmae.item())
            loss_mse.backward()
            optimizer.step()

        # Validate after each epoch
        model.eval()
        val_loss_mse = []
        val_loss_rmae = []
        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)[:, -pred_len:, :]
                batch_xy = torch.cat([batch_x, batch_y], dim=1)
                output = model(batch_x)
                if ft:
                    output = output[:, -pred_len:, f_dim:]
                    batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                    loss_mse = criterion_mse(output, batch_y)
                    loss_rmae = criterion_rmae(output, batch_y)
                else:
                    output = output[:, :, f_dim:]
                    loss_mse = criterion_mse(output, batch_xy)
                    loss_rmae = criterion_rmae(output, batch_xy)

                val_loss_mse.append(loss_mse.item())
                val_loss_rmae.append(loss_rmae.item())

        mean_val_loss_mse = np.mean(val_loss_mse)
        mean_val_loss_rmae = np.mean(val_loss_rmae)

        scheduler.step(mean_val_loss_mse)

        new_lr = optimizer.param_groups[0]["lr"]
        if current_lr != new_lr:
            print(f"Learning Rate changed to: {new_lr}")
            current_lr = new_lr

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss_mse": np.mean(train_loss_mse),
                "train_loss_rmae": np.mean(train_loss_rmae),
                "val_loss_mse": mean_val_loss_mse,
                "val_loss_rmae": mean_val_loss_rmae,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch: {epoch+1} \t Train MSE: {np.mean(train_loss_mse):.4f} \t Val MSE: {mean_val_loss_mse:.4f}"
        )

        # print(
        #     f"Epoch: {epoch+1} \t Train MSE: {np.mean(train_loss_mse):.4f} \t Train RMAE: {np.mean(train_loss_rmae):.4f} \t Val MSE: {mean_val_loss_mse:.4f} \t Val RMAE: {mean_val_loss_rmae:.4f}"
        # )

        # Early stopping
        if mean_val_loss_mse < best_val_loss - min_delta:
            best_val_loss = mean_val_loss_mse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    model, test_mse = test(
        model=model,
        test_loader=test_loader,
        f_dim=f_dim,
        device=device,
        pred_len=pred_len,
        ft=True,
    )

    return model, test_mse


def test(
    model: nn.Module,
    test_loader: DataLoader,
    f_dim: int,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    pred_len=360,
    ft=True,
    args=None,
):

    model.to(device)
    criterion_mse = nn.MSELoss()
    criterion_rmae = RMAE

    with torch.no_grad():
        model.eval()
        test_loss_mse = []
        test_loss_rmae = []
        i = 0
        for i, (batch_x, batch_y, *_) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)[:, -pred_len:, :]
            batch_xy = torch.cat([batch_x, batch_y], dim=1).to(device)
            output = model(batch_x)
            if ft:
                output = output[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                loss_mse = criterion_mse(output, batch_y)
                loss_rmae = criterion_rmae(output, batch_y)
            else:
                output = output[:, :, f_dim:]
                loss_mse = criterion_mse(output, batch_xy)
                loss_rmae = criterion_rmae(output, batch_xy)

            test_loss_mse.append(loss_mse.item())
            test_loss_rmae.append(loss_rmae.item())
            i += 1
            if i == 50 and args.model == "ARIMA":
                break

        wandb.log(
            {
                "test_loss_mse": np.mean(test_loss_mse),
                "test_loss_rmae": np.mean(test_loss_rmae),
            }
        )

        print(
            f"Test loss MSE: {np.mean(test_loss_mse):.4f}, Test loss RMAE: {np.mean(test_loss_rmae):.4f}"
        )

    return model, np.mean(test_loss_mse)
