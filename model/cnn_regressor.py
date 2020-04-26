import torch.nn as nn
import torchvision.models as models


class CNNRegressor(nn.Module):
    def __init__(self, output_size, feature_extraction=False):
        super(CNNRegressor, self).__init__()

        original_model = models.__dict__["vgg16"](pretrained=True)

        self.features = original_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_size),
        )

        if feature_extraction:
            # Freeze features weights
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
