from torch import nn
import torch

class ClassifierModelV1(nn.Module):
    def __init__(self, input_shape=1, output_shape=1):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(51200, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, output_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x