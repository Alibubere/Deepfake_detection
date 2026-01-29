from torch.utils.data import Dataset
import numpy as np
import torch

class MemapDataset(Dataset):
    """
    Memory map dataset
    """
    def __init__(self,bin_path,labels_path,shape,transform=None):

        self.images = np.memmap(bin_path,dtype="unit8",mode="r",shape=shape)
        self.labels_tensor = torch.from_numpy(np.load(labels_path)).long()
        self.transform = transform


    def __len__(self):
        return len(self.labels_tensor)
    
    def __getitem__(self, index):
        
        img = torch.from_numpy(self.images[index].copy().transpose(2,0,1)).float().div_(225.0)

        if self.transform:
            img = self.transform(img)

        return img , self.labels_tensor[index]        