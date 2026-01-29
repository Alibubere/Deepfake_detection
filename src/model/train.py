import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch_amp(model:nn.Module,dataloader,optimizer:torch.optim.Optimizer,device:torch.device,loss_fn,scaler):
    """Training with automatic mixed precision - uses fp16 for speed"""

    model.train()
    train_loss , train_acc = 0.0,0.0

    for X , y in tqdm(dataloader,desc="Training",leave=False):

        X , y = X.to(device,non_blocking = True) , y.to(device,non_blocking = True)

        with torch.amp.autocast("cuda"):
            y_preds = model(X)
            loss = loss_fn(y_preds,y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        y_pred_class = torch.argmax(y_preds,dim=1)
        train_acc += (y_pred_class == y).sum().item() / y.size(0)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss , train_acc
