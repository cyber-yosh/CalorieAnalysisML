import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

class CalorieDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, missing_log_path="missing_dishes.txt"):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_data = []

        raw_data = pd.read_csv(csv_file, header=None)

        with open(missing_log_path, "w") as log_file:
            for _, row in raw_data.iterrows():
                dish_id = str(row[0])
                img_path = os.path.join(self.root_dir, dish_id, "rgb.png")
                if os.path.exists(img_path):
                    self.valid_data.append((dish_id, row[1]))
                else:
                    log_file.write(f"{dish_id}\n")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        dish_id, calorie = self.valid_data[idx]
        img_path = os.path.join(self.root_dir, dish_id, "rgb.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([calorie], dtype=torch.float32)
        return image, label

class MultiImageCalorieDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, missing_log_path="missing_dishes.txt"):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_data = []

        raw_data = pd.read_csv(csv_file, header=None)

        with open(missing_log_path, "w") as log_file:
            for _, row in raw_data.iterrows():
                dish_id = str(row[0])
                image_paths = {
                    "rgb": os.path.join(self.root_dir, dish_id, "rgb.png"),
                    "depth_raw": os.path.join(self.root_dir, dish_id, "depth_raw.png"),
                    "depth_color": os.path.join(self.root_dir, dish_id, "depth_color.png")
                }

                # Check all files exist
                if not all(os.path.exists(p) for p in image_paths.values()):
                    log_file.write(f"{dish_id} — missing file\n")
                    continue

                # Try to open all files
                try:
                    for path in image_paths.values():
                        with Image.open(path) as img:
                            img.verify()  # Check if it's a valid image
                except (UnidentifiedImageError, OSError):
                    log_file.write(f"{dish_id} — invalid/corrupted image\n")
                    continue

                # All good → include in dataset
                self.valid_data.append((dish_id, row[1]))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        dish_id, calorie = self.valid_data[idx]

        def load_image(filename):
            path = os.path.join(self.root_dir, dish_id, filename)
            img = Image.open(path).convert("RGB")
            return self.transform(img) if self.transform else img

        rgb = load_image("rgb.png")
        depth_raw = load_image("depth_raw.png")
        depth_color = load_image("depth_color.png")

        label = torch.tensor([calorie], dtype=torch.float32)

        return rgb, depth_raw, depth_color, label

class AllNutrientsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, missing_log_path="missing_dishes.txt"):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_data = []

        raw_data = pd.read_csv(csv_file, header=None)

        with open(missing_log_path, "w") as log_file:
            for _, row in raw_data.iterrows():
                dish_id = str(row[0])
                img_path = os.path.join(self.root_dir, dish_id, "rgb.png")
                if os.path.exists(img_path):
                    # Grab all 5 labels: calories, mass, fat, carbs, protein
                    labels = row[1:6].values.astype("float32")
                    self.valid_data.append((dish_id, labels))
                else:
                    log_file.write(f"{dish_id}\n")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        dish_id, labels = self.valid_data[idx]
        img_path = os.path.join(self.root_dir, dish_id, "rgb.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return image, label_tensor