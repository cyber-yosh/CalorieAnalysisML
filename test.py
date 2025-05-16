import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN, MultiCNN, AllNutrientsCNN
from dataset import CalorieDataset, MultiImageCalorieDataset, AllNutrientsDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

model = AllNutrientsCNN()
model.load_state_dict(torch.load("allnutrients_model.pth"))
model.eval()

from PIL import Image
from torchvision import transforms
import torch

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define same transform used in training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load image
img = Image.open("test.png").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 64, 64]

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.cpu().numpy().flatten()

# Print results
nutrients = ["Calories", "Mass", "Fat", "Carbs", "Protein"]
for name, value in zip(nutrients, prediction):
    print(f"{name}: {round(value, 2)}")
