import logging
import os
import yaml
from src.data_prep.optimized_dataset import get_optimized_dataset, get_transform


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

    # Save paths
    save_paths = config["save_paths"]
    gray_path = save_paths["gray"]
    clahe_path = save_paths["clahe"]
    edges_path = save_paths["edges"]
    rgb_path = save_paths["RGB"]

    # Full paths
    gray_full_path = os.path.join(save_dir, gray_path)
    clahe_full_path = os.path.join(save_dir, clahe_path)
    edges_full_path = os.path.join(save_dir, edges_path)
    rgb_full_path = os.path.join(save_dir, rgb_path)

    transform = get_transform()

    full_dataset_gray = get_optimized_dataset(
        root_dir, gray_full_path, mode="gray", transform=transform
    )
    full_dataset_clahe = get_optimized_dataset(
        root_dir, clahe_full_path, mode="clahe", transform=transform
    )
    full_dataset_edges = get_optimized_dataset(
        root_dir, save_path=edges_full_path, mode="edges", transform=transform
    )
    full_dataset_RGB = get_optimized_dataset(
        root_dir, rgb_full_path, mode="RGB", transform=transform
    )


if __name__ == "__main__":
    main()
