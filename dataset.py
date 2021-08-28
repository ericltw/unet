import torch
from torch.utils.data import Dataset
import glob
import os
from torchvision.io import read_image

class FVCDataset(Dataset):
    def __init__(self):
        # return a list of paths.
        self.images_path = glob.glob(os.path.join('data/train/original/*.png'))

    def __getitem__(self, index):
        image_path = self.images_path[index]
        label_path = image_path.replace('original', 'label')

        image: torch.Tensor = read_image(image_path)
        label: torch.Tensor = read_image(label_path)

        return image, label
    
    def __len__(self):
        return len(self.images_path)