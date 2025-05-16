import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
            torch.nn.ReLU()
        )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x   
    
    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return x

class MultiCNN(nn.Module):
    def __init__(self):
        super(MultiCNN, self).__init__()

        # Shared CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Flattened feature size: 64 x 8 x 8 = 4096 (for input 64x64)
        # After 3 branches: 4096 * 3 = 12288
        self.fc1 = nn.Linear(4096 * 3, 512)
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward_one(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x1, x2, x3):
        f1 = self.forward_one(x1)
        f2 = self.forward_one(x2)
        f3 = self.forward_one(x3)

        x = torch.cat([f1, f2, f3], dim=1)  # shape: [batch_size, 12288]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict(self, x1, x2, x3):
        with torch.no_grad():
            return self.forward(x1, x2, x3)

class AllNutrientsCNN(torch.nn.Module):
    def __init__(self):
        super(AllNutrientsCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(512, 5),  # 5 outputs: calories, mass, fat, carbs, protein
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return x


