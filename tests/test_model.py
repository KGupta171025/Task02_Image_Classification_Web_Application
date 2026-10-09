"""
Unit Tests for PyTorch Model Classifier and Metrics Utilities.
"""

import io
import numpy as np
import pytest
import torch
from PIL import Image

from core.config import CIFAR10_CLASSES
from core.metrics import compute_top_k_accuracy
from core.model import build_model, get_classifier, get_inference_transforms


def create_dummy_image_bytes(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Helper to generate in-memory dummy image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_build_model_mobilenet():
    """Verify MobileNetV2 builds with adapted 10-class head."""
    model = build_model("mobilenet_v2", num_classes=10, pretrained_backbone=False)
    assert isinstance(model, torch.nn.Module)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_build_model_resnet50():
    """Verify ResNet50 builds with adapted 10-class head."""
    model = build_model("resnet50", num_classes=10, pretrained_backbone=False)
    assert isinstance(model, torch.nn.Module)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_build_model_invalid():
    """Verify error raised on unknown model architecture."""
    with pytest.raises(ValueError):
        build_model("unsupported_cnn", num_classes=10)


def test_classifier_predict_structure():
    """Verify inference pipeline returns valid Top-5 predictions."""
    classifier = get_classifier("mobilenet_v2")
    img_bytes = create_dummy_image_bytes(150, 150)
    result = classifier.predict(img_bytes, top_k=5)

    assert result["model_name"] == "mobilenet_v2"
    assert "inference_time_ms" in result
    assert result["inference_time_ms"] > 0
    assert "top_1" in result
    assert len(result["top_predictions"]) == 5

    # Check top_1 keys
    top_1 = result["top_1"]
    assert top_1["rank"] == 1
    assert top_1["class_name"] in CIFAR10_CLASSES
    assert 0.0 <= top_1["confidence"] <= 1.0

    # Ensure probabilities are sorted descending
    confs = [p["confidence"] for p in result["top_predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_compute_top_k_accuracy():
    """Verify top-k accuracy calculation helper."""
    y_true = [0, 1, 2]
    # Prediction probabilities for 3 samples across 3 classes
    y_probs = [
        [0.8, 0.1, 0.1],  # top-1 is 0 -> match
        [0.3, 0.6, 0.1],  # top-1 is 1 -> match
        [0.2, 0.3, 0.5],  # top-1 is 2 -> match
    ]
    top_1 = compute_top_k_accuracy(y_true, y_probs, k=1)
    assert top_1 == 100.0

    # Sample where top-1 is incorrect but top-2 is correct
    y_probs_partial = [
        [0.8, 0.1, 0.1],  # true 0, top-1 is 0
        [0.7, 0.3, 0.0],  # true 1, top-1 is 0, top-2 is 1
        [0.6, 0.3, 0.1],  # true 2, top-1 is 0, top-2 is 1, top-3 is 2
    ]
    top_1_partial = compute_top_k_accuracy(y_true, y_probs_partial, k=1)
    top_2_partial = compute_top_k_accuracy(y_true, y_probs_partial, k=2)
    assert top_1_partial == pytest.approx(33.33, 0.1)
    assert top_2_partial == pytest.approx(66.67, 0.1)
