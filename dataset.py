import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

class FVCDataset(Dataset):
    def __init__(self):
        # return a list of paths.
        self.images_path = glob.glob(os.path.join('data/train/original/*.png'))
        # TODO: Check Tseng's method
        self.resize = transforms.Compose([
            transforms.Resize(160)
        ])

    def __getitem__(self, index):
        image_path = self.images_path[index]
        label_path = image_path[:24] + '.png'
        label_path = label_path.replace('original', 'label')

        image: torch.Tensor = read_image(image_path)
        label: torch.Tensor = read_image(label_path)
        
        transformed_image: torch.Tensor = self.resize(image)
        transformed_label: torch.Tensor = self.resize(label)

        # 陽春的二值化
        transformed_label = transformed_label / label.max()
        transformed_label[transformed_label >= 0.7] = 1
        transformed_label[transformed_label < 0.7] = 0

        # print('transformed_label.shape')
        # print(transformed_label.shape)
        # print('transformed_label')
        # print(transformed_label)

        return transformed_image, transformed_label
    
    def __len__(self):
        return len(self.images_path)