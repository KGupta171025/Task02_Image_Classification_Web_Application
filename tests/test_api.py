"""
Unit and Integration Tests for FastAPI Endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from core.config import CIFAR10_CLASSES

client = TestClient(app)


def create_test_image(format_name: str = "JPEG", size=(64, 64), color="blue") -> bytes:
    """Generates synthetic image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def test_root_endpoint():
    """Verify root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "health_check" in data
    assert "predict_endpoint" in data


def test_health_check_endpoint():
    """Verify /api/health returns healthy status and model listing."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mobilenet_v2" in data["available_models"]
    assert "resnet50" in data["available_models"]
    assert data["num_classes"] == 10


def test_models_list_endpoint():
    """Verify /api/models returns supported architectures and CIFAR-10 classes."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "mobilenet_v2" in data["supported_models"]
    assert "resnet50" in data["supported_models"]
    assert data["classes"] == CIFAR10_CLASSES


def test_metrics_endpoint():
    """Verify /api/metrics returns accuracy statistics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "top_1_accuracy" in data
    assert "top_5_accuracy" in data


def test_predict_endpoint_valid_image():
    """Verify /api/predict processes image and returns Top-5 predictions."""
    img_bytes = create_test_image("JPEG")
    files = {"file": ("test_sample.jpg", img_bytes, "image/jpeg")}
    data = {"model_name": "mobilenet_v2", "top_k": 5}

    response = client.post("/api/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["model_name"] == "mobilenet_v2"
    assert "inference_time_ms" in res
    assert res["top_1"]["rank"] == 1
    assert len(res["top_predictions"]) == 5

    # Check top_1 validity
    assert res["top_1"]["class_name"] in CIFAR10_CLASSES
    assert 0.0 <= res["top_1"]["confidence"] <= 1.0


def test_predict_endpoint_png_format():
    """Verify /api/predict handles PNG images."""
    img_bytes = create_test_image("PNG")
    files = {"file": ("test_sample.png", img_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_predict_invalid_model_name():
    """Verify /api/predict rejects unsupported model architectures."""
    img_bytes = create_test_image("JPEG")
    files = {"file": ("test_sample.jpg", img_bytes, "image/jpeg")}
    data = {"model_name": "vgg16"}

    response = client.post("/api/predict", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid model_name" in response.json()["detail"]


def test_predict_unsupported_media_type():
    """Verify /api/predict rejects invalid file extensions/MIME types."""
    files = {"file": ("document.pdf", b"%PDF-1.4 dummy", "application/pdf")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 415


def test_predict_empty_file():
    """Verify /api/predict rejects 0-byte files."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_predict_corrupted_image():
    """Verify /api/predict handles non-image bytes gracefully."""
    files = {"file": ("corrupt.jpg", b"NOT_A_VALID_IMAGE_BINARY_DATA", "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert "decoded" in response.json()["detail"]
