"""
Application Configuration and Constants.
Defines paths, model hyperparameters, dataset labels, and runtime settings.
"""

from pathlib import Path
from typing import Dict, List

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLE_IMAGES_DIR = ASSETS_DIR / "sample_images"
SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"

# Ensure runtime directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Classes: CIFAR-10
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Normalization constants (Standard ImageNet / Transfer Learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image input size for deep CNN backbones
IMAGE_SIZE = 224

# Model configurations
MODEL_CONFIGS: Dict[str, dict] = {
    "mobilenet_v2": {
        "display_name": "MobileNetV2 (Optimized & Fast)",
        "architecture": "MobileNetV2",
        "description": "Lightweight inverted residual architecture, ideal for low-latency edge & web inference.",
        "params": "~3.5 Million",
        "weights_file": WEIGHTS_DIR / "mobilenetv2_cifar10.pth",
        "num_classes": 10,
        "default": True,
    },
    "resnet50": {
        "display_name": "ResNet50 (Deep Residual Network)",
        "architecture": "ResNet50",
        "description": "50-layer deep residual network with bottleneck blocks for rich feature representations.",
        "params": "~25.6 Million",
        "weights_file": WEIGHTS_DIR / "resnet50_cifar10.pth",
        "num_classes": 10,
        "default": False,
    },
}

# API Configuration
API_TITLE = "Nexoramind Image Classification API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-grade CNN Image Classification API supporting ResNet50 and MobileNetV2 "
    "with Top-5 prediction confidence scoring and real-time inference latency tracking."
)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]
DEFAULT_TOP_K = 5
