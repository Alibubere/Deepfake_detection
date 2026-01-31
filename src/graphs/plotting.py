import matplotlib.pyplot as plt
from typing import Dict
import os
import logging


def plot_model_performance(history: Dict, plot_dir, file_name: str):

    os.makedirs(plot_dir, exist_ok=True)

    if not file_name.endswith((".png", ".jpg", ".gif", ".jpeg")):
        raise ValueError(f"File name should ends with .png , .jpg etc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["test_loss"], "r-", label="Test Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training vs Test Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        "b-",
        label="Train Acc",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        [a * 100 for a in history["test_acc"]],
        "r-",
        label="Test Acc",
        linewidth=2,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training vs Test Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, file_name), dpi=150, bbox_inches="tight")
    logging.info(f"Training curves saved to {plot_dir}/{file_name}")
    plt.show()
