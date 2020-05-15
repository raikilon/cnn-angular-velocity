import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
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


        self.targets = np.zeros((2, len(self.images)))

        # Loop over all images that have been marked as object
        for i in range(len(object_flags)):
            # normalization between 0 and 1 (1 is given when the object is at most 5 centimetres away)
            center_left = 1 - max((raw_target[i]["center_left"] - min_distance) / (sensor_range - min_distance),
                                  0)
            center_right = 1 - max((raw_target[i]["center_right"] - min_distance) / (sensor_range - min_distance),
                                   0)
            center = 1 - max((raw_target[i]["center"] - min_distance) / (sensor_range - min_distance), 0)

            # compute T and C for current image
            T = np.around(np.dot([center_left, center_right], [1, -1]), decimals=1)
            C = np.around(center, decimals=1)
            target = [T, C]
            idx = object_flags[i]
            # Assign the same target to the current picture and its 4 previous ones
            self.targets[:, idx] = target
            if idx - 1 >= 0:
                self.targets[:, idx - 1] = target
            if idx - 2 >= 0:
                self.targets[:, idx - 2] = target
            if idx - 3 >= 0:
                self.targets[:, idx - 3] = target
            if idx-4 >= 0:
                self.targets[:, idx - 4] = target



        for img_idx in pitfall_flags:
            # For each image marked as pitfalls assign C to 1 to it and its previous 4 pictures
            for j in range(0, 8):
                self.targets[1, img_idx - j] = 1

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
        target = np.array(self.targets[:,idx], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
