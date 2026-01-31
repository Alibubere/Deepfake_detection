import os
import logging
from torch.utils.data import TensorDataset
import torch
from src.data_prep.Dataset import ImageDatasetwithCV
from torchvision.transforms import transforms, Resize
from src.data_prep.MemapDataset import MemmapDataset
import numpy as np
from tqdm import tqdm


def get_optimized_dataset(
    raw_dataset_path, save_prefix, save_dir ,mode="RGB", target_size=(224, 224), transform=None
):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    bin_path = f"{save_prefix}_{mode}_images.bin"
    label_path = f"{save_prefix}_{mode}_labels.npy"
    meta_path = f"{save_prefix}_{mode}_meta.pt"

    bin_path_full = os.path.join(save_dir,bin_path)
    label_path_full = os.path.join(save_dir,label_path)
    meta_path_full = os.path.join(save_dir,meta_path)

    if (
        os.path.exists(bin_path_full)
        and os.path.exists(label_path_full)
        and os.path.exists(meta_path_full)
    ):
        logging.info(f"Loading from preprocessed data from {bin_path_full}...")
        meta = torch.load(meta_path_full, weights_only=True)
        return MemmapDataset(
            bin_path_full, label_path_full, shape=meta["shape"], transform=transform
        )

    logging.info(f"Preprocessing images to memmap format ({mode})....")

    save_transform = Resize(target_size)
    raw_set = ImageDatasetwithCV(raw_dataset_path, mode=mode, transform=None)
    num_samples = len(raw_set)
    img_shape = (num_samples, target_size[0], target_size[1], 3)

    fp = np.memmap(bin_path_full, dtype=np.uint8, mode="w+", shape=img_shape)

    all_labels = []

    for i in tqdm(range(num_samples), desc="Processing Image"):
        img_tensor, label = raw_set[i]
        img_resized = save_transform(img_tensor)
        img_np = (img_resized.permute(1, 2, 0).numpy() * 225).astype(np.uint8)
        fp[i] = img_np
        all_labels.append(label)

    fp.flush()
    np.save(label_path_full, np.array(all_labels))
    torch.save({"shape": img_shape}, meta_path_full)

    logging.info(f"Preprocessing complete! Saved to {save_prefix}")
    return MemmapDataset(bin_path_full, label_path_full, shape=img_shape, transform=transform)


def get_train_transform():
    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ]
    )

    return transform


def get_test_transform():
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform
