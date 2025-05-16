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

def evaluate_simple():
    threshold = 0.2

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = CalorieDataset(csv_file="test.csv", root_dir="nutrition5k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("simplecnn_calorie_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(outputs.flatten())
            all_labels.extend(labels.flatten())

    correct = 0
    total = len(all_labels)

    within_threshold_flags = []  # for binary matrix

    # For calorie bin confusion matrix
    bin_width = 100
    actual_bins = []
    predicted_bins = []

    for pred, label in zip(all_preds, all_labels):
        within_threshold = abs(pred - label) <= threshold * label
        within_threshold_flags.append(1 if within_threshold else 0)

        if within_threshold:
            correct += 1

        # Binning for heatmap confusion matrix
        true_bin = int(round(label / bin_width) * bin_width)
        pred_bin = int(round(pred / bin_width) * bin_width)
        actual_bins.append(true_bin)
        predicted_bins.append(pred_bin)

    accuracy = correct / total if total > 0 else 0.0
    print(f"✅ Accuracy within {int(threshold*100)}% of ground truth: {accuracy:.4f} ({correct}/{total})")

    average_error = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    print(f"\n📏 Average Absolute Error: {average_error:.2f} kcal")

    # Binary confusion matrix
    y_true_binary = [1] * total
    y_pred_binary = within_threshold_flags

    binary_cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
    print("\n📊 Binary Confusion Matrix (based on threshold):")
    print("Rows = actual, Columns = predicted")
    print(pd.DataFrame(binary_cm, index=["Actual Correct", "Actual Wrong"], columns=["Predicted Correct", "Predicted Wrong"]))

    # Calorie bin confusion matrix
    labels_sorted = sorted(set(actual_bins + predicted_bins))
    cm = confusion_matrix(actual_bins, predicted_bins, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    print("\n🔢 Calorie Confusion Matrix (binned, rows = actual, columns = predicted):")
    print(cm_df)

def train_simple():
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Transform (resize to match model input)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = CalorieDataset(csv_file="train.csv", root_dir="nutrition5k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}:")
                print(f"  Predicted: {outputs[0].item():.2f}")
                print(f"  Label:     {labels[0].item():.2f}")
                print(f"  Loss:      {loss.item():.2f}")

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "simplecnn_calorie_model.pth")
    print("Model saved as simplecnn_calorie_model.pth")

def train_multi():
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = MultiImageCalorieDataset(csv_file="train.csv", root_dir="nutrition5k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model = MultiCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (rgb, depth_raw, depth_color, labels) in enumerate(dataloader):
            rgb = rgb.to(device)
            depth_raw = depth_raw.to(device)
            depth_color = depth_color.to(device)
            labels = labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(rgb, depth_raw, depth_color)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb.size(0)

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}")
                print(f"  Predicted: {outputs[0].item():.2f}")
                print(f"  Label:     {labels[0].item():.2f}")
                print(f"  Loss:      {loss.item():.2f}")

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "multicnn_calorie_model.pth")
    print("Model saved as multicnn_calorie_model.pth")

def eval_multi():
    threshold = 0.2

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = MultiImageCalorieDataset(csv_file="test.csv", root_dir="nutrition5k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = MultiCNN().to(device)
    model.load_state_dict(torch.load("multicnn_calorie_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, depth_raw, depth_color, labels in dataloader:
            rgb = rgb.to(device)
            depth_raw = depth_raw.to(device)
            depth_color = depth_color.to(device)
            labels = labels.to(device)

            outputs = model(rgb, depth_raw, depth_color).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(outputs.flatten())
            all_labels.extend(labels.flatten())

    correct = 0
    total = len(all_labels)
    within_threshold_flags = []

    bin_width = 100
    actual_bins = []
    predicted_bins = []

    for pred, label in zip(all_preds, all_labels):
        within_threshold = abs(pred - label) <= threshold * label
        within_threshold_flags.append(1 if within_threshold else 0)

        if within_threshold:
            correct += 1

        true_bin = int(round(label / bin_width) * bin_width)
        pred_bin = int(round(pred / bin_width) * bin_width)
        actual_bins.append(true_bin)
        predicted_bins.append(pred_bin)

    accuracy = correct / total if total > 0 else 0.0
    avg_error = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

    print(f"✅ Accuracy within {int(threshold*100)}% of ground truth: {accuracy:.4f} ({correct}/{total})")
    print(f"📏 Average Absolute Error: {avg_error:.2f} kcal")

    y_true_binary = [1] * total
    y_pred_binary = within_threshold_flags

    binary_cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
    print("\n📊 Binary Confusion Matrix (based on threshold):")
    print("Rows = actual, Columns = predicted")
    print(pd.DataFrame(binary_cm, index=["Actual Correct", "Actual Wrong"], columns=["Predicted Correct", "Predicted Wrong"]))

    labels_sorted = sorted(set(actual_bins + predicted_bins))
    cm = confusion_matrix(actual_bins, predicted_bins, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    print("\n🔢 Calorie Confusion Matrix (binned, rows = actual, columns = predicted):")
    print(cm_df)


def train_all():
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = AllNutrientsDataset(csv_file="train.csv", root_dir="nutrition5k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model = AllNutrientsCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if i % 10 == 0:
                pred = [round(float(v), 2) for v in outputs[0].detach().cpu()]
                true = [round(float(v), 2) for v in labels[0].detach().cpu()]
                print(f"Epoch {epoch+1}, Batch {i}:")
                print(f"  Predicted: {pred}")
                print(f"  Label:     {true}")
                print(f"  Loss:      {loss.item():.2f}")

                

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "allnutrients_model.pth")
    print("Model saved as allnutrients_model.pth")

def eval_all():
    import numpy as np
    from sklearn.metrics import accuracy_score

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = AllNutrientsDataset(csv_file="train.csv", root_dir="nutrition5k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model
    model = AllNutrientsCNN().to(device)
    model.load_state_dict(torch.load("allnutrients_model.pth", map_location=device))
    model.eval()

    threshold = 0.2  # 20% relative error
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if i % 10 == 0:
                pred = outputs[0].cpu().numpy()
                true = labels[0].cpu().numpy()
                rel_errors = np.abs((pred - true) / (true + 1e-8))  # avoid divide-by-zero
                print(f"Batch {i}:")
                print(f"  Predicted: {[round(v, 2) for v in pred]}")
                print(f"  Label:     {[round(v, 2) for v in true]}")
                print(f"  Rel. Error: {[round(e, 2) for e in rel_errors]}")

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    abs_errors = np.abs(preds - labels)
    rel_errors = np.abs((preds - labels) / (labels + 1e-8))  # relative error

    n_nutrients = preds.shape[1]
    nutrient_names = ["Calories", "Mass", "Fat", "Carbs", "Protein"][:n_nutrients]

    print("\n--- Evaluation Summary ---")
    for i in range(n_nutrients):
        mae = np.mean(abs_errors[:, i])
        accuracy = np.mean(rel_errors[:, i] <= threshold)
        print(f"{nutrient_names[i]}:")
        print(f"  Mean Absolute Error: {mae:.3f}")
        print(f"  Accuracy (within 20%): {accuracy * 100:.2f}%")
        print()

    # Confusion-like matrix (pass/fail per nutrient)
    binary_matrix = (rel_errors <= threshold).astype(int)
    print("--- Pass Matrix (1 = within 20%, 0 = not) ---")
    print("Sample | " + " | ".join(nutrient_names))
    print("-" * (9 + 6 * n_nutrients))
    for idx in range(min(10, len(binary_matrix))):
        row = "   {}    | ".format(idx) + " | ".join(str(int(v)) for v in binary_matrix[idx])
        print(row)

    overall_mae = np.mean(abs_errors)
    print(f"\nOverall MAE: {overall_mae:.4f}")


if __name__ == "__main__":
    #simple_train()
    #simple_evaluate()
    #train_multi()
    #eval_multi()
   #train_all()
    eval_all()
