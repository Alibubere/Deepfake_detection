import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch_amp(
    model: nn.Module,
    dataloader:torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn,
    scaler,
    transform=None
):
    """Training with automatic mixed precision - uses fp16 for speed"""

    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in tqdm(dataloader, desc="Training"):

        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if transform:
            X = transform(X)

        with torch.amp.autocast("cuda"):
            y_preds = model(X)
            loss = loss_fn(y_preds, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        y_pred_class = torch.argmax(y_preds, dim=1)
        train_acc += (y_pred_class == y).sum().item() / y.size(0)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_one_epoch(model: nn.Module, dataloader:torch.utils.data.DataLoader, device: torch.device, loss_fn):
    """Validation/testing"""
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    for X, y in tqdm(dataloader, desc="Testing"):

        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        y_pred_class = torch.argmax(y_pred, dim=1)

        test_loss += loss.item()
        test_acc += (y_pred_class == y).sum().item() / y.size(0)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc