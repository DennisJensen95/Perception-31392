import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import numpy as np

class Net(nn.Module):
    def __init__(self, input_shape, classes, learning_rate):
        super(Net, self).__init__()


        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.ReLU()
        )

        out_features = self.conv_out_features(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(out_features, 512),
            nn.ReLU(),
            nn.Linear(512, classes)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def conv_out_features(self, shape):
        o = self.feature(torch.zeros(1, *shape))
        return int(np.prod(o.size()))



