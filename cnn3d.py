import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# class Motor3DCNN(nn.Module):
#     def __init__(self):
#         super(Motor3DCNN, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool3d(2, 2)
#         self.fc1 = nn.Linear(32 * 16 * 16 * 16, 128)  # Adjust based on input shape
#         self.fc2 = nn.Linear(128, 1)  # Binary classification (motor present or not)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = self.pool(nn.ReLU()(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Flatten
#         x = nn.ReLU()(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x

class LightMotor3DCNN(nn.Module):
    def __init__(self):
        super(LightMotor3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

        # Compute output size dynamically (use a dummy input)
        dummy_input = torch.zeros(1, 1, 16, 256, 256)  # (batch, channels, depth, height, width)
        with torch.no_grad():
            dummy_out = self.pool(F.relu(self.conv3(self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input)))))))))
            flatten_size = dummy_out.numel() // dummy_out.shape[0]  # Total features per batch

        self.fc = nn.Linear(flatten_size, 64)  # Adjusted based on computed size

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))

        x = self.conv2(x)
        x = self.pool(F.relu(x))

        x = self.conv3(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, start_dim=1)  # Flatten all except batch
        x = self.fc(x)
        return x