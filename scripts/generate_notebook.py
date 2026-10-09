"""
Generates the comprehensive training and evaluation Jupyter Notebook.
"""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title & Intro
cells.append(nbf.v4.new_markdown_cell("""# 🧠 Nexoramind AI Engineer Internship — Task 02
## Deep Learning Image Classification System with CNN Models (MobileNetV2 / ResNet50)

**Author:** Krishna Gupta  
**Project:** Task 02 - Image Classification Web Application  
**Tech Stack:** Python 3.12, PyTorch, TorchVision, FastAPI, Streamlit, Scikit-learn, Matplotlib  
**Dataset:** CIFAR-10 (10 Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)

---

### 📌 Objectives
1. **Model Architecture:** Construct and fine-tune CNN models (MobileNetV2 and ResNet50) using Transfer Learning.
2. **Data Pipeline:** Build PyTorch `DataLoader` with augmentations (random flips, rotations, ImageNet normalization).
3. **Training & Optimization:** Train model using CrossEntropyLoss and AdamW optimizer with Cosine Annealing scheduler.
4. **Comprehensive Evaluation:** Calculate **Top-1** and **Top-5** Accuracy, per-class Precision/Recall/F1, and Confusion Matrix.
5. **Model Checkpointing:** Export optimized weights for integration with FastAPI backend and Streamlit Web Application.
"""))

# Cell 2: Setup & Environment
cells.append(nbf.v4.new_markdown_cell("""### 1. Setup & Environment Verification"""))
cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"TorchVision Version: {torchvision.__version__}")
print(f"Execution Device: {device}")
"""))

# Cell 3: Dataset & Data Augmentations
cells.append(nbf.v4.new_markdown_cell("""### 2. Dataset Pipeline & Data Augmentations
We load the CIFAR-10 dataset and apply data augmentations to enhance model generalization:
- Resizing to 224x224 (standard CNN input dimension for ImageNet pre-trained backbones)
- Random Horizontal Flip
- Random Rotation (±10 degrees)
- Normalization using ImageNet Mean `[0.485, 0.456, 0.406]` and Std `[0.229, 0.224, 0.225]`
"""))

cells.append(nbf.v4.new_code_cell("""IMAGE_SIZE = 224
BATCH_SIZE = 64

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "./data"
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Classes: {CIFAR10_CLASSES}")
"""))

# Cell 4: DataLoaders
cells.append(nbf.v4.new_code_cell("""train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Visualize a batch of training images
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    return np.clip(img, 0, 1)

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, ax in enumerate(axes.flat):
    ax.imshow(denormalize(images[idx]))
    ax.set_title(CIFAR10_CLASSES[labels[idx]], fontsize=11, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""))

# Cell 5: Model Architecture Setup
cells.append(nbf.v4.new_markdown_cell("""### 3. Model Architecture Setup (Transfer Learning)
We leverage **MobileNetV2** (lightweight, inverted residual blocks) and adapt the final classification linear layer from 1,000 ImageNet categories to the 10 target CIFAR-10 categories.
"""))

cells.append(nbf.v4.new_code_cell("""def build_mobilenet_v2(num_classes=10):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    # Replace final classification head
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

model = build_mobilenet_v2(num_classes=10).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model Architecture: MobileNetV2")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
"""))

# Cell 6: Training Pipeline
cells.append(nbf.v4.new_markdown_cell("""### 4. Loss Function, Optimizer & Learning Rate Scheduler"""))
cells.append(nbf.v4.new_code_cell("""criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
epochs = 3
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
"""))

# Cell 7: Training Loop
cells.append(nbf.v4.new_markdown_cell("""### 5. Training Loop"""))
cells.append(nbf.v4.new_code_cell("""history = {"train_loss": [], "train_acc": []}

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(epoch_acc)
    
    print(f"Epoch [{epoch}/{epochs}] ({time.time() - start_time:.1f}s) - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
"""))

# Cell 8: Quantitative Evaluation
cells.append(nbf.v4.new_markdown_cell("""### 6. Quantitative Evaluation: Top-1 and Top-5 Accuracy
According to internship task requirements, we evaluate both **Top-1** and **Top-5** accuracy on the held-out test dataset.
"""))

cells.append(nbf.v4.new_code_cell("""model.eval()
all_targets = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_targets.extend(targets.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

# Calculate Top-1 and Top-5 Accuracy
def compute_top_k(y_true, y_probs, k=5):
    correct = 0
    for true_label, p in zip(y_true, y_probs):
        top_k_indices = np.argsort(p)[-k:][::-1]
        if true_label in top_k_indices:
            correct += 1
    return (correct / len(y_true)) * 100.0

top_1_acc = compute_top_k(all_targets, all_probs, k=1)
top_5_acc = compute_top_k(all_targets, all_probs, k=5)

print("=" * 45)
print(f"📊 TOP-1 ACCURACY: {top_1_acc:.2f}%")
print(f"⭐ TOP-5 ACCURACY: {top_5_acc:.2f}%")
print("=" * 45)
"""))

# Cell 9: Per-class Classification Report
cells.append(nbf.v4.new_markdown_cell("""### 7. Per-Class Precision, Recall, and F1-Score"""))
cells.append(nbf.v4.new_code_cell("""report = classification_report(all_targets, all_preds, target_names=CIFAR10_CLASSES, digits=4)
print(report)
"""))

# Cell 10: Confusion Matrix Heatmap
cells.append(nbf.v4.new_markdown_cell("""### 8. Confusion Matrix Visualization"""))
cells.append(nbf.v4.new_code_cell("""cm = confusion_matrix(all_targets, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(9, 7), dpi=300)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CIFAR10_CLASSES,
    yticklabels=CIFAR10_CLASSES,
    cbar_kws={"label": "Normalized Proportion"}
)
plt.title("Normalized Confusion Matrix - CIFAR-10 Classification", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../assets/confusion_matrix.png", dpi=300)
plt.show()
"""))

# Cell 11: Qualitative Predictions with Top-5 Confidence
cells.append(nbf.v4.new_markdown_cell("""### 9. Sample Test Predictions with Top-5 Confidence Distributions"""))
cells.append(nbf.v4.new_code_cell("""# Inspect sample prediction
sample_idx = 0
sample_tensor = test_dataset[sample_idx][0].unsqueeze(0).to(device)
true_label = CIFAR10_CLASSES[test_dataset[sample_idx][1]]

with torch.no_grad():
    output = model(sample_tensor)
    probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

top5_idx = np.argsort(probs)[-5:][::-1]
top5_labels = [CIFAR10_CLASSES[i] for i in top5_idx]
top5_scores = [probs[i] * 100.0 for i in top5_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(denormalize(test_dataset[sample_idx][0]))
ax1.set_title(f"Ground Truth: {true_label}", fontweight="bold")
ax1.axis("off")

y_pos = np.arange(len(top5_labels))
ax2.barh(y_pos, top5_scores, color="#0284c7")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top5_labels)
ax2.invert_yaxis()
ax2.set_xlabel("Confidence (%)")
ax2.set_title("Top-5 Predicted Confidence", fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# Cell 12: Checkpoint Export
cells.append(nbf.v4.new_markdown_cell("""### 10. Checkpoint Export for Production Deployment
We serialize the fine-tuned model state dictionary into `weights/mobilenetv2_cifar10.pth` for real-time serving in FastAPI and Streamlit.
"""))

cells.append(nbf.v4.new_code_cell("""weights_path = Path("../weights/mobilenetv2_cifar10.pth")
weights_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "model_name": "mobilenet_v2",
    "classes": CIFAR10_CLASSES,
    "top_1_accuracy": top_1_acc,
    "top_5_accuracy": top_5_acc,
}, weights_path)

print(f"✅ Model weights saved successfully to: {weights_path.resolve()}")
"""))

nb.cells = cells

out_notebook = Path("D:/Code_Files/Projects/Nexoramind/Task_02-Image_Classification_Web_Application/notebooks/cifar10_classification_training_evaluation.ipynb")
with open(out_notebook, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook at {out_notebook}")
