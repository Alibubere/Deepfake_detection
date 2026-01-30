from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import train_test_split


def get_split_data(full_dataset, test_size=0.2):

    labels = [label for _, label in full_dataset]
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    return train_dataset, test_dataset
