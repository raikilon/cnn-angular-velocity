from torch.utils.data import Dataset
import torch
import os
import numpy as np
from skimage import io
from torchvision import transforms


class ObstaclePositionDataset(Dataset):
    def __init__(self, images_dir, target_file):
        self.image_dir = images_dir
        self.images = sorted(os.listdir(images_dir))
        self.targets = np.load(target_file)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(os.path.join(self.image_dir, self.images[idx]))
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
