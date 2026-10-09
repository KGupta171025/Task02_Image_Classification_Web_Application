"""
Pydantic Schema Definitions for API Requests and Responses.
Provides strong type validation and OpenAPI specification documentation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Individual prediction entry."""
    rank: int = Field(..., description="Prediction rank from 1 to 5")
    class_id: int = Field(..., description="Numeric class index in CIFAR-10")
    class_name: str = Field(..., description="Human-readable category label")
    confidence: float = Field(..., description="Softmax confidence score (0.0 to 1.0)")
    confidence_percent: str = Field(..., description="Formatted percentage string")


class ImageMetadata(BaseModel):
    """Metadata regarding the processed image."""
    filename: str
    content_type: str
    size_bytes: int


class PredictionResponse(BaseModel):
    """Payload returned by POST /api/predict."""
    success: bool = True
    model_name: str
    architecture: str
    device: str
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    image_info: ImageMetadata
    top_1: PredictionItem
    top_predictions: List[PredictionItem] = Field(..., description="Top-5 predicted classes with confidence")


class HealthResponse(BaseModel):
    """Payload returned by GET /api/health."""
    status: str
    version: str
    device: str
    available_models: List[str]
    default_model: str
    num_classes: int


class ModelDetail(BaseModel):
    """Detailed model architecture specifications."""
    display_name: str
    architecture: str
    description: str
    params: str
    num_classes: int
    default: bool


class ModelsListResponse(BaseModel):
    """Payload returned by GET /api/models."""
    supported_models: Dict[str, ModelDetail]
    dataset: str
    classes: List[str]


class MetricsResponse(BaseModel):
    """Payload returned by GET /api/metrics."""
    model_name: str
    dataset: str
    total_test_samples: int
    top_1_accuracy: float
    top_5_accuracy: float
    confusion_matrix: List[List[int]]
    confusion_matrix_plot_url: str
    classification_report: Dict


class ErrorResponse(BaseModel):
    """Standardized API Error Response."""
    success: bool = False
    error: str
    detail: str
    status_code: int
