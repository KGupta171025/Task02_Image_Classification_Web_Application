"""
Evaluation & Benchmarking Script.
Loads trained model, benchmarks Top-1/Top-5 accuracy on CIFAR-10 test set,
and exports the confusion matrix and evaluation metrics report.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CIFAR10_CLASSES, MODEL_CONFIGS
from core.metrics import generate_evaluation_summary
from core.model import get_classifier
from scripts.train import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")


def evaluate_model(model_name: str = "mobilenet_v2", test_samples: int = 1000):
    """Benchmarks the classifier on the CIFAR-10 test partition."""
    logger.info("Evaluating model '%s' with %d test samples...", model_name, test_samples)
    classifier = get_classifier(model_name)

    data_dir = PROJECT_ROOT / "data"
    _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        subset_test_size=test_samples,
    )

    all_targets = []
    all_preds = []
    all_probs = []

    classifier.model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(classifier.device)
            outputs = classifier.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    summary = generate_evaluation_summary(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        class_names=CIFAR10_CLASSES,
        model_name=model_name,
    )

    logger.info("=== Evaluation Results ===")
    logger.info("Top-1 Accuracy: %.2f%%", summary["top_1_accuracy"])
    logger.info("Top-5 Accuracy: %.2f%%", summary["top_5_accuracy"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 classification model")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    evaluate_model(model_name=args.model, test_samples=args.samples)
