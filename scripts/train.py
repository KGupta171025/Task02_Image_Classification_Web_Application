"""
Model Training and Fine-Tuning Script.
Trains MobileNetV2 / ResNet50 on CIFAR-10, benchmarks Top-1 and Top-5 accuracy,
generates the confusion matrix, and extracts sample test images.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    ASSETS_DIR,
    CIFAR10_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SAMPLE_IMAGES_DIR,
    WEIGHTS_DIR,
)
from core.metrics import generate_evaluation_summary
from core.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    subset_train_size: int = 0,
    subset_test_size: int = 0,
) -> Tuple[DataLoader, DataLoader, torchvision.datasets.CIFAR10]:
    """Prepares CIFAR-10 train and test loaders with data augmentations."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    logger.info("Loading CIFAR-10 dataset (downloading if required)...")
    train_set = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    if subset_train_size > 0 and subset_train_size < len(train_set):
        logger.info("Using subset of %d training samples for rapid fine-tuning", subset_train_size)
        indices = torch.randperm(len(train_set))[:subset_train_size]
        train_set = Subset(train_set, indices)

    if subset_test_size > 0 and subset_test_size < len(test_set):
        logger.info("Using subset of %d test samples for evaluation", subset_test_size)
        indices = torch.randperm(len(test_set))[:subset_test_size]
        test_set = Subset(test_set, indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    raw_test_set = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
    )

    return train_loader, test_loader, raw_test_set


def save_sample_images(raw_test_dataset, output_dir: Path = SAMPLE_IMAGES_DIR):
    """Exports representative sample images per CIFAR-10 class for easy interactive UI testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_classes = set()
    logger.info("Extracting representative sample images into %s...", output_dir)

    for idx in range(len(raw_test_dataset)):
        img, label = raw_test_dataset[idx]
        class_name = CIFAR10_CLASSES[label]
        if class_name not in saved_classes:
            filename = output_dir / f"{class_name}_sample.png"
            # Upscale 32x32 CIFAR image slightly for visual clarity in UI
            upscaled = img.resize((224, 224), Image.Resampling.BILINEAR)
            upscaled.save(filename, format="PNG")
            saved_classes.add(class_name)
            if len(saved_classes) == len(CIFAR10_CLASSES):
                break

    logger.info("Saved %d sample images successfully.", len(saved_classes))


def train_and_evaluate(
    model_name: str = "mobilenet_v2",
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 64,
    subset_train: int = 2500,
    subset_test: int = 1000,
    save_weights: bool = True,
):
    """Main training, validation, and evaluation routine."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting training run with %s on %s", model_name, device)

    data_dir = PROJECT_ROOT / "data"
    train_loader, test_loader, raw_test_set = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        subset_train_size=subset_train,
        subset_test_size=subset_test,
    )

    # Save sample test images for the UI
    save_sample_images(raw_test_set)

    # Instantiate model with pre-trained ImageNet backbone
    model = build_model(model_name=model_name, num_classes=len(CIFAR10_CLASSES), pretrained_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop
    logger.info("Fine-tuning %s for %d epochs...", model_name, epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(
                    "Epoch [%d/%d] Step [%d/%d] Loss: %.4f | Train Acc: %.2f%%",
                    epoch,
                    epochs,
                    batch_idx + 1,
                    len(train_loader),
                    running_loss / total,
                    100.0 * correct / total,
                )

        scheduler.step()

    # Evaluation on Test Set
    logger.info("Evaluating %s on test set...", model_name)
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # Generate complete evaluation metrics and confusion matrix plot
    summary = generate_evaluation_summary(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        class_names=CIFAR10_CLASSES,
        model_name=model_name,
    )

    logger.info("Evaluation Complete!")
    logger.info("Top-1 Accuracy: %.2f%%", summary["top_1_accuracy"])
    logger.info("Top-5 Accuracy: %.2f%%", summary["top_5_accuracy"])
    logger.info("Confusion matrix saved to %s", summary["confusion_matrix_plot"])

    # Save model weights
    if save_weights:
        weights_file = WEIGHTS_DIR / f"{model_name.replace('-', '_')}_cifar10.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "classes": CIFAR10_CLASSES,
            "top_1_accuracy": summary["top_1_accuracy"],
            "top_5_accuracy": summary["top_5_accuracy"],
        }, weights_file)
        logger.info("Model weights checkpoint saved to %s", weights_file)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model on CIFAR-10")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-train", type=int, default=2000, help="Train subset size (0 for full)")
    parser.add_argument("--subset-test", type=int, default=600, help="Test subset size (0 for full)")
    args = parser.parse_args()

    train_and_evaluate(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
    )
