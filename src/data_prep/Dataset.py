from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2
import torch


class ImageDatasetwithCV(Dataset):

    def __init__(self, root_dir, mode="RGB", transform=None):
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.transform = transform

        self.class_to_idx = {}
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):

            self.class_to_idx[cls] = idx

            cls_dir = os.path.join(root_dir, cls)

            for file in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.mode == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[..., None]

        elif self.mode == "clahe":
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            l = clahe.apply(l)
            img = cv2.merge((l, a, b))
            img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

        elif self.mode == "edges":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.Canny(gray, 100, 200)[..., None]

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        if img.ndim == 2:
            img = img[..., None]

        img = img.permute(2, 0, 1)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        if self.transform:
            img = self.transform(img)

        return img, label
