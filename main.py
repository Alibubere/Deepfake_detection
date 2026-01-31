import logging
import os
import yaml
import gc
import torch
from src.data_prep.optimized_dataset import (
    get_optimized_dataset,
    get_train_transform,
    get_test_transform,
)
from src.model.train_loop import train
from src.model.train_utils import get_optimizer, get_resnet18_model
from src.data_prep.dataset_split import get_split_data
from src.data_prep.dataloader import get_dataloader
from src.graphs.plotting import plot_model_performance


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
    logging.info("Logging initialized successfully")


def main():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Directories and Config
    root_dir = config["root_dir"]
    save_dir = config["save_dir"]
    save_prefix = config["save_prefix"]
    os.makedirs(save_dir, exist_ok=True)

    train_config = config["training"]
    num_epochs = train_config["num_epochs"]
    batch_size = train_config["batch_size"]
    num_workers = train_config["num_workers"]
    lr = train_config["lr"]
    weight_decay = train_config["weight_decay"]
    resume = train_config["resume"]

    model_config = config["model"]
    model_dir = model_config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    graph_config = config["graph"]
    graph_dir = graph_config["graph_dir"]
    os.makedirs(graph_dir, exist_ok=True)

    # Transforms
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # Define the training phases to run sequentially
    phases = [
        {
            "mode": "RGB",
            "latest": os.path.join(model_dir, model_config["rgb_latest"]),
            "best": os.path.join(model_dir, model_config["rgb_best"]),
            "plot": graph_config["rgb_plot"],
        },
        {
            "mode": "clahe",
            "latest": os.path.join(model_dir, model_config["clahe_latest"]),
            "best": os.path.join(model_dir, model_config["clahe_best"]),
            "plot": graph_config["clahe_plot"],
        },
        {
            "mode": "edges",
            "latest": os.path.join(model_dir, model_config["edges_latest"]),
            "best": os.path.join(model_dir, model_config["edges_best"]),
            "plot": graph_config["edges_plot"],
        },
        {
            "mode": "gray",
            "latest": os.path.join(model_dir, model_config["gray_latest"]),
            "best": os.path.join(model_dir, model_config["gray_best"]),
            "plot": graph_config["gray_plot"],
        },
    ]

    # Shared Loss and Scaler
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for phase in phases:
        mode = phase["mode"]
        logging.info(f"\n{'='*30}\nSTARTING PHASE: {mode.upper()}\n{'='*30}")

        # 1. Initialize fresh model and optimizer for each phase
        model = get_resnet18_model(device=device)
        optimizer = get_optimizer(model=model, lr=lr, weight_decay=weight_decay)

        # 2. Load Data for THIS phase only
        full_dataset = get_optimized_dataset(
            root_dir,
            save_prefix,
            save_dir=save_dir,
            mode=mode,
            transform=test_transform,
        )
        train_dataset, test_dataset = get_split_data(full_dataset)

        logging.info(
            f"{mode} - Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
        )

        # 3. Create DataLoaders
        train_loader = get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True,
        )
        test_loader = get_dataloader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

        # 4. Train
        history = train(
            resume=resume,
            model=model,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            latest_path=phase["latest"],
            best_path=phase["best"],
            loss_fn=loss_fn,
            scaler=scaler,
            transform=train_transform
        )

        # 5. Plot performance
        plot_model_performance(history, plot_dir=graph_dir, file_name=phase["plot"])

        # 6. CRITICAL CLEANUP FOR WINDOWS
        del (
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
            full_dataset,
            model,
            optimizer,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Finished Phase: {mode.upper()}. Resources cleared.")

    logging.info("All training pipelines completed successfully.")


if __name__ == "__main__":
    main()
