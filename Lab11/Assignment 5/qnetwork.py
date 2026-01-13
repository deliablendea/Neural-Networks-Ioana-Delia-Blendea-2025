import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_shape, actions):
        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=3),
            nn.ReLU6(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU6()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.get_conv_output_size(input_shape), 256),
            nn.ReLU6(),
            nn.Linear(256, actions)
        )

    def get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))