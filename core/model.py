"""
PyTorch Model Management, Preprocessing, and Inference Engine.
Handles MobileNetV2 and ResNet50 architectures, weight loading, and Top-K predictions.
"""

import io
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from core.config import (
    CIFAR10_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_CONFIGS,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


def get_inference_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Standard inference preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(
    model_name: str,
    num_classes: int = len(CIFAR10_CLASSES),
    pretrained_backbone: bool = True,
) -> nn.Module:
    """
    Constructs the CNN model backbone and adapts the classification head.
    
    Args:
        model_name: 'mobilenet_v2' or 'resnet50'
        num_classes: Target number of classification categories (10 for CIFAR-10)
        pretrained_backbone: Whether to load pre-trained ImageNet weights for transfer learning
    """
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model name '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")


class ModelClassifier:
    """
    Inference manager responsible for model lifecycle, tensor transformations,
    and computing top-K prediction distributions.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classes = CIFAR10_CLASSES
        self.num_classes = len(self.classes)
        self.transform = get_inference_transforms()
        self.weights_path = weights_path or MODEL_CONFIGS[model_name]["weights_file"]

        logger.info("Initializing %s on %s...", self.model_name, self.device)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> nn.Module:
        """Instantiates architecture and loads weights if present on disk."""
        model = build_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained_backbone=True,
        )

        # Check if fine-tuned checkpoint exists
        if self.weights_path and self.weights_path.exists():
            logger.info("Loading fine-tuned weights from %s", self.weights_path)
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                logger.warning("Could not load weights from %s: %s. Using backbone defaults.", self.weights_path, e)
        else:
            logger.info("No checkpoint found at %s. Using ImageNet pre-trained backbone.", self.weights_path)

        return model

    def preprocess_image(self, image_data: Union[bytes, Image.Image]) -> torch.Tensor:
        """
        Converts raw bytes or PIL Image to normalized batch tensor [1, 3, 224, 224].
        """
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            raise ValueError("Expected image bytes or PIL.Image instance")

        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image_data: Union[bytes, Image.Image],
        top_k: int = 5,
    ) -> Dict:
        """
        Executes forward pass and returns top-K predictions with probabilities and timings.
        """
        start_time = time.perf_counter()

        tensor = self.preprocess_image(image_data)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # Extract top-K scores and indices
        top_k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        predictions: List[Dict] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append({
                "rank": len(predictions) + 1,
                "class_id": int(idx),
                "class_name": self.classes[idx],
                "confidence": float(round(prob, 4)),
                "confidence_percent": f"{prob * 100.0:.2f}%",
            })

        return {
            "model_name": self.model_name,
            "architecture": MODEL_CONFIGS[self.model_name]["architecture"],
            "device": str(self.device),
            "inference_time_ms": round(latency_ms, 2),
            "top_1": predictions[0] if predictions else None,
            "top_predictions": predictions,
        }


# Model Registry / Cache for instant re-use across API and UI
_MODEL_CACHE: Dict[str, ModelClassifier] = {}


def get_classifier(model_name: str = "mobilenet_v2") -> ModelClassifier:
    """Returns a cached ModelClassifier instance for the specified model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}")

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = ModelClassifier(model_name=model_name)

    return _MODEL_CACHE[model_name]
