import torch
import torch.nn as nn
import torch.nn.functional as F

# this is for the commit


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, 5)
        self.conv2 = nn.Conv2d(50, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(20*20*128, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 20*20*128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
