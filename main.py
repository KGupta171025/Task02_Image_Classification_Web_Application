"""
Main Entry Point for Nexoramind Task 02.
Supports running the FastAPI backend, Streamlit frontend, or training pipeline.
"""

import argparse
import subprocess
import sys


def run_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Launches the FastAPI backend with Uvicorn."""
    import uvicorn
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


def run_frontend(port: int = 8501):
    """Launches the Streamlit frontend."""
    cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", str(port)]
    subprocess.run(cmd)


def run_train():
    """Runs the model training and evaluation script."""
    from scripts.train import train_and_evaluate
    train_and_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexoramind Image Classification Application")
    parser.add_argument(
        "--mode",
        type=str,
        default="api",
        choices=["api", "frontend", "train"],
        help="Service mode to execute: 'api', 'frontend', or 'train'",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.mode == "api":
        run_api(host=args.host, port=args.port)
    elif args.mode == "frontend":
        run_frontend(port=args.port if args.port != 8000 else 8501)
    elif args.mode == "train":
        run_train()
