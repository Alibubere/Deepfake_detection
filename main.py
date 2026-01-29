import logging
import os
import yaml
from src.data_prep.optimized_dataset import (
    get_optimized_dataset,
    get_train_transform,
    get_test_transform,
)


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


if __name__ == "__main__":
    main()
