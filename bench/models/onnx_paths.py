"""Utilities for managing ONNX model paths."""

from pathlib import Path
from typing import List, Optional


def get_models_dir() -> Path:
    """Get the models directory path."""
    # Assumes this file is in bench/models/
    return Path(__file__).parent.parent.parent / "models"


def get_onnx_path(model_name: str) -> Path:
    """
    Get the path for an ONNX model file.
    
    Args:
        model_name: Name of the model (without .onnx extension)
        
    Returns:
        Path to the ONNX model file
    """
    models_dir = get_models_dir()
    
    # Add .onnx extension if not present
    if not model_name.endswith(".onnx"):
        model_name = f"{model_name}.onnx"
    
    return models_dir / model_name


def list_available_models() -> List[str]:
    """
    List all available ONNX models.
    
    Returns:
        List of model names (without .onnx extension)
    """
    models_dir = get_models_dir()
    
    if not models_dir.exists():
        return []
    
    onnx_files = list(models_dir.glob("*.onnx"))
    return [f.stem for f in onnx_files]


def ensure_models_dir() -> Path:
    """
    Ensure the models directory exists.
    
    Returns:
        Path to the models directory
    """
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
