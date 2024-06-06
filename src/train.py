import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import datetime
from rich import print

def RMAE(output, target):
    return torch.sqrt(torch.mean(torch.abs(output - target)))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 1000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    pred_len=360,
    features="M",
    ft=True,
    lr=5e-4,
    args=None,
):
    wandb.init(
        project="ssl_fts",
        entity="ssl_fts",
        name=f'{args.dataset}_feat_{args.features}_pred_{args.pred_len}_label_{args.label_len}_seq_{args.seq_len}-time_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        mode="online" if args.use_wandb else "disabled",
    )
    model.to(device)
    criterion_mse = nn.MSELoss()
    criterion_rmae = RMAE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, threshold=1e-4
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

        scheduler.step(np.mean(train_loss_mse))

        new_lr = optimizer.param_groups[0]["lr"]
        if current_lr != new_lr:
            print(f"Learning Rate changed to: {new_lr}")
            current_lr = new_lr

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss_mse": np.mean(train_loss_mse),
                "train_loss_rmae": np.mean(train_loss_rmae),
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch: {epoch+1} \t MSE: {np.mean(train_loss_mse):.4f} \t RMAE: {np.mean(train_loss_rmae):.4f}"
        )

    with torch.no_grad():
        model.eval()
        test_loss_mse = []
        test_loss_rmae = []
        for batch_x, batch_y, *_ in test_loader:
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

            test_loss_mse.append(loss_mse.item())
            test_loss_rmae.append(loss_rmae.item())

        wandb.log(
            {
                "test_loss_mse": np.mean(test_loss_mse),
                "test_loss_rmae": np.mean(test_loss_rmae),
            }
        )

        print(
            f"Test loss MSE: {np.mean(test_loss_mse)}, Test loss RMAE: {np.mean(test_loss_rmae)}"
        )

    return model
