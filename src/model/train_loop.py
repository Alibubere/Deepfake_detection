from src.model.train import (
    train_one_epoch_amp,
    test_one_epoch,
    save_checkpoint,
    load_checkpoint,
)
import torch
import os
import logging


def train(
    resume: bool,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    latest_path: str,
    best_path: str,
    loss_fn,
    scaler,
):
    """
    Main training loop for a PyTorch model with support for resuming from checkpoints
    and mixed-precision training.

    Args:
        resume (bool): Whether to resume training from the latest checkpoint.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Total number of epochs to train.
        device (torch.device): Device to run training on (CPU/GPU).
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test/validation data.
        latest_path (str): Path to save the latest checkpoint after each epoch.
        best_path (str): Path to save the best-performing model checkpoint.
        loss_fn (callable): Loss function used for training and evaluation.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed-precision training.

    Workflow:
        1. Optionally resumes training from the latest checkpoint if available.
        2. Initializes tracking lists for training and testing losses/accuracies.
        3. Iterates through epochs:
            - Trains the model for one epoch using mixed precision.
            - Evaluates the model on the test set.
            - Logs training and testing metrics.
            - Saves the best model checkpoint if test accuracy improves.
            - Saves the latest checkpoint at the end of each epoch.
        4. Logs final best test accuracy after training completes.
        5. Returns a dictionary containing lists of losses and accuracies for both
           training and testing across all epochs.

    Returns:
        dict: {
            "train_loss": list of training losses per epoch,
            "train_acc": list of training accuracies per epoch,
            "test_loss": list of test losses per epoch,
            "test_acc": list of test accuracies per epoch
        }
    """

    start_epoch = 1
    best_test_acc = 0

    if resume and os.path.exists(latest_path):

        try:
            model, optimizer, start_epoch, best_test_acc = load_checkpoint(
                model, optimizer, device, latest_path
            )

            logging.info(
                f"Resumed from epoch {start_epoch-1}, best acc: {best_test_acc*100:.2f}%"
            )

        except Exception as e:
            logging.exception(
                f"⚠ Could not load checkpoint: {e}. Starting from scratch."
            )
            start_epoch = 1
            best_test_acc = 0

    else:
        logging.info("Starting training from scratch...")

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(start_epoch, num_epochs + 1):

        logging.info(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch_amp(
            model, train_loader, optimizer, device, loss_fn, scaler
        )

        test_loss, test_acc = test_one_epoch(model, test_loader, device, loss_fn)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logging.info(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")

        if test_acc > best_test_acc:
            improvement = (test_acc - best_test_acc) * 100

            logging.info(
                f"Test accuracy improved by {improvement:.2f}%! Saving best model..."
            )

            best_test_acc = test_acc

            save_checkpoint(model, optimizer, epoch, best_path, best_test_acc)

        save_checkpoint(model, optimizer, epoch, latest_path, best_test_acc)

    logging.info(f"Training complete! Best test accuracy: {best_test_acc*100:.2f}%")

    return {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses,
        "test_acc": test_accs,
    }
