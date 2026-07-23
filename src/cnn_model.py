import torch
import torch.nn as nn

# this model predicts one box per image
# it is small and easy to train
class CustomCNN(nn.Module):
    def __init__(self, max_objects=1):
        super().__init__()
        self.max_objects = max_objects
        # this is the simple cnn layout
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, max_objects * 5),
        )

    # this sends the image through the network
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, self.max_objects, 5)
        return x

