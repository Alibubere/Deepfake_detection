import os
import logging
from torch.utils.data import TensorDataset
import torch
from src.data_prep.Dataset import ImageDatasetwithCV
from torchvision.transforms import transforms

def get_optimized_dataset(raw_dataset_path, save_path, mode="RGB", transform=None):
    if os.path.exists(save_path):

        logging.info(f"Loading processed data from {save_path}")

        data = torch.load(save_path)

        return TensorDataset(data["images"], data["labels"])

    else:
        logging.info(f"No saved data found. processing with Opencv {mode}")

        processed_set = ImageDatasetwithCV(
            raw_dataset_path, mode=mode, transform=transform
        )

        all_images = []
        all_labels = []

        for i in range(len(processed_set)):

            img_tensor, label = processed_set[i]
            all_images.append(img_tensor)
            all_labels.append(label)

        images_tensor = torch.stack(all_images)
        labels_tensor = torch.tensor(all_labels)

        logging.info(f"Saving processed data to {save_path}")
        torch.save({"images": images_tensor, "labels": labels_tensor}, save_path)

        return TensorDataset(images_tensor, labels_tensor)



def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225]),
    ]
    )

    return transform
