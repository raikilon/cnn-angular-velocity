from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms


class ObstaclePositionDataset(Dataset):
    def __init__(self, images_dir):
        # image store every 4s
        self.image_dir = os.path.join(images_dir, "imgs")
        self.images = os.listdir(self.image_dir)
        self.images = sorted(self.images, key=lambda x: int(os.path.splitext(x)[0]))
        # raw targets stored 3 times within max (0.12) and min proximity sensor threshold (0.01)
        raw_target = np.load(os.path.join(images_dir, "sensor_data.npy"), allow_pickle=True)
        # object flags stored at min proximity sensor threshold(0.01)
        object_flags = np.load(os.path.join(images_dir, "object_flags.npy"), allow_pickle=True)
        # pitfall flags stored at max groundn threshold (0.11)
        pitfall_flags = np.load(os.path.join(images_dir, "pitfall_flags.npy"), allow_pickle=True)

        self.targets = []
        min_distance = 0.05
        sensor_range = 0.12

        ####### Begin: Implementation of extra part, not relying on artificial simulation data
        self.targets = np.zeros((3, len(self.images)))

        # target values for object detection
        for i in range(len(object_flags)):
            for j in range(0, 3):
                idx = 3 * i - j
                center_left = 1 - max((raw_target[idx]["center_left"] - min_distance) / (sensor_range - min_distance),
                                      0)
                center_right = 1 - max((raw_target[idx]["center_right"] - min_distance) / (sensor_range - min_distance),
                                       0)
                center = 1 - max((raw_target[idx]["center"] - min_distance) / (sensor_range - min_distance), 0)

                T = np.around(np.dot([center_left, center_right], [1, -1]), decimals=1)
                C = np.around(center, decimals=1)
                target = [T, C, 0]

                flag = object_flags[i - j]
                self.targets[:, flag] = target

        # target values for pitfall detection
        step = 1
        for img_idx in pitfall_flags:
            for j in range(0, 3):
                self.targets[2, img_idx - j] = step - 1/3

        ####### End: Implementation of extra part, not relying on artificial simulation data

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
