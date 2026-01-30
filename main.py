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


if __name__ == "__main__":
    main()
