import torch.nn as nn
import torchvision.models as models


class CNNRegressor(nn.Module):
    def __init__(self, output_size, feature_extraction=False):
        super(CNNRegressor, self).__init__()

        # Load VGG16 with pretrained weights on ImageNet
        original_model = models.__dict__["vgg16"](pretrained=True)

        # use vgg16 weights (without last fully connected layers)
        self.features = original_model.features
        # Create new head for the network with the correct output size
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_size),
        )

        # freeze gradient for transfer learning
        if feature_extraction:
            # Freeze features weights
            for p in self.features.parameters():
                p.requires_grad = False

    # forward function of the network
    def forward(self, x):
        f = self.features(x)
        # reshape features for classifier
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
