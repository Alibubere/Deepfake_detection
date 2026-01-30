import logging
import os
import yaml
from src.data_prep.optimized_dataset import (
    get_optimized_dataset,
    get_train_transform,
    get_test_transform,
)
from src.model.train_loop import train
from src.model.train_utils import get_optimizer, get_resnet18_model
from src.data_prep.dataset_split import get_split_data
from src.data_prep.dataloader import get_dataloader
import torch


def setup_logging():

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_name = "Pipeline.log"
    full_path = os.path.join(log_dir, file_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(full_path), logging.StreamHandler()],
    )

    logging.info("Logging initialize successfully")


def main():

    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Root dir
    root_dir = config["root_dir"]

    # Save dir
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Save Prefix
    save_prefix = config["save_prefix"]

    # Training config
    train_config = config["training"]
    num_epochs = train_config["num_epochs"]
    batch_size = train_config["batch_size"]
    num_workers = train_config["num_workers"]
    lr = train_config["lr"]
    weight_decay = train_config["weight_decay"]
    resume = train_config["resume"]

    # Model Config
    model_config = config["model"]
    model_dir = model_config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # Gray paths
    gray_latest = model_config["gray_latest"]
    gray_latest_full = os.path.join(model_dir, gray_latest)
    gray_best = model_config["gray_best"]
    gray_best_full = os.path.join(model_dir, gray_best)

    # Clahe paths
    clahe_latest = model_config["clahe_latest"]
    clahe_latest_full = os.path.join(model_dir, clahe_latest)
    clahe_best = model_config["clahe_best"]
    clahe_best_full = os.path.join(model_dir, clahe_best)

    # Edges paths
    edges_latest = model_config["edges_latest"]
    edges_latest_full = os.path.join(model_dir, edges_latest)
    edges_best = model_config["edges_best"]
    edges_best_full = os.path.join(model_dir, edges_best)

    # RGB paths
    rgb_latest = model_config["rgb_latest"]
    rgb_latest_full = os.path.join(model_dir, rgb_latest)
    rgb_best = model_config["rgb_best"]
    rgb_best_full = os.path.join(model_dir, rgb_best)

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    full_dataset_gray = get_optimized_dataset(
        root_dir, save_prefix, save_dir=save_dir, mode="gray", transform=test_transform
    )
    full_dataset_clahe = get_optimized_dataset(
        root_dir, save_prefix, save_dir=save_dir, mode="clahe", transform=test_transform
    )
    full_dataset_edges = get_optimized_dataset(
        root_dir, save_prefix, save_dir=save_dir, mode="edges", transform=test_transform
    )
    full_dataset_RGB = get_optimized_dataset(
        root_dir, save_prefix, save_dir=save_dir, mode="RGB", transform=test_transform
    )

    train_dataset_gray, test_dataset_gray = get_split_data(full_dataset_gray)
    train_dataset_clahe, test_dataset_clahe = get_split_data(full_dataset_clahe)
    train_dataset_edges, test_dataset_egdes = get_split_data(full_dataset_edges)
    train_dataset_rgb, test_dataset_rgb = get_split_data(full_dataset_RGB)

    logging.info(f"Train samples: {len(train_dataset_gray)}")
    logging.info(f"Test samples: {len(test_dataset_gray)}")

    train_loader_gray = get_dataloader(
        train_dataset_gray,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    test_loader_gray = get_dataloader(
        test_dataset_gray,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=True,
    )

    train_loader_clahe = get_dataloader(
        train_dataset_clahe,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    test_loader_clahe = get_dataloader(
        test_dataset_clahe,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=True,
    )

    train_loader_edges = get_dataloader(
        train_dataset_edges,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    test_loader_edges = get_dataloader(
        test_dataset_egdes,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=True,
    )
    train_loader_rgb = get_dataloader(
        train_dataset_rgb,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    test_loader_rgb = get_dataloader(
        test_dataset_rgb,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=True,
    )

    model = get_resnet18_model(device=device)
    optimizer = get_optimizer(model=model, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss()

    history_rgb = train(
        resume=resume,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        train_loader=train_loader_rgb,
        test_loader=test_loader_rgb,
        latest_path=rgb_latest_full,
        best_path=rgb_best_full,
        loss_fn=loss_fn,
        scaler=scaler,
    )

    history_clahe = train(
        resume=resume,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        train_loader=train_loader_clahe,
        test_loader=test_loader_clahe,
        latest_path=clahe_latest_full,
        best_path=clahe_best_full,
        loss_fn=loss_fn,
        scaler=scaler,
    )

    history_edges = train(
        resume=resume,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        train_loader=train_loader_edges,
        test_loader=test_loader_edges,
        latest_path=edges_latest_full,
        best_path=edges_best_full,
        loss_fn=loss_fn,
        scaler=scaler,
    )

    history_gray = train(
        resume=resume,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        train_loader=train_loader_gray,
        test_loader=test_loader_gray,
        latest_path=gray_latest_full,
        best_path=gray_best_full,
        loss_fn=loss_fn,
        scaler=scaler,
    )


if __name__ == "__main__":
    main()
