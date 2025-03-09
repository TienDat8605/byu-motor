import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LightMotor3DCNN(nn.Module):
    def __init__(self):
        super(LightMotor3DCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Reduces size by 2x
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Reduces size by 2x
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Reduces size by 2x
        )

        # Compute correct input size for FC layer
        self.fc_input_size = 256 * 2 * 16 * 16  # Based on output size after conv layers

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x