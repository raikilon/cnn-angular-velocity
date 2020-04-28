from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms


class ObstaclePositionDataset(Dataset):
    def __init__(self, images_dir):
        self.image_dir = os.path.join(images_dir, "imgs")
        self.images = os.listdir(self.image_dir)
        self.images = sorted(self.images, key=lambda x: int(os.path.splitext(x)[0]))
        raw_target = np.load(os.path.join(images_dir, "sensor_data.npy"), allow_pickle=True)
        self.targets = []
        min_distance = 0.5
        sensor_range = 2
        for elem in raw_target:
            center_left = 1 - max((elem["center_left"] - min_distance) / (sensor_range - min_distance), 0)
            center_right = 1 - max((elem["center_right"] - min_distance) / (sensor_range - min_distance), 0)
            center = 1 - max((elem["center"] - min_distance) / (sensor_range - min_distance), 0)

            T = np.around(np.dot([center_left, center_right], [1, -1]), decimals=1)
            C = np.around(center, decimals=1)
            target = [T, C]
            self.targets.append(target)

        # np.intersect1d(np.where(np.array(self.targets)[:, 0] == 0), np.where(np.array(self.targets)[:, 1] == 0))
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, self.images[idx]), 'rb') as f:
            img = Image.open(f)
            image = img.convert('RGB')
        target = np.array(self.targets[idx], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
