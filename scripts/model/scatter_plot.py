from torch.utils.data import DataLoader
from cnn_regressor import CNNRegressor
from torchvision import transforms
from mark_and_save_dataset import ObstaclePositionDataset
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import random_split

# Init CNN model
model = CNNRegressor(2, False)
checkpoint = torch.load("pitfalls.tar", map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
del checkpoint
torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ObstaclePositionDataset(os.path.join("data", "train"))
val_size = int((len(dataset) / 100) * 20)
_, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
device = torch.device('cpu')
outputs = []
targets = []
for input_val, target_val in loader:
    target_val = target_val.to(device=device)
    targets.extend(target_val.detach().cpu().numpy())
    input_val = input_val.to(device=device)
    with torch.no_grad():
        output = model(input_val)
        output = output.detach().cpu().numpy()
        outputs.extend(output)

outputs = np.array(outputs)
targets = np.array(targets)
x = outputs[:, 1]
y = targets[:, 1]
plt.scatter(x, y, alpha=0.5)
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.show()
