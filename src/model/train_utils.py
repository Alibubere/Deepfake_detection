import torch
from torch import nn
from torchvision.models import resnet18


def get_optimizer(model: nn.Module, lr, weight_decay):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer


def get_resnet18_model(device: torch.device):
    model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device=device)

    return model


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path, best_test_acc
):
    """Save model checkpoint"""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_test_acc": best_test_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, path
):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 1) + 1
    best_test_acc = checkpoint.get("best_test_acc", 0)

    return model, optimizer, start_epoch, best_test_acc
