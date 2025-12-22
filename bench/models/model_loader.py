"""Utilities for loading and preparing models."""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn

class FastGELU(nn.Module):
    """
    Fast GELU approximation using tanh.
    More fusion-friendly than the cubic version.
    """
    def forward(self, x):
        # Original: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Simplified: 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

def replace_gelu(m: nn.Module):
    # Import here to avoid issues if transformers is not installed
    try:
        from transformers.activations import GELUActivation
    except ImportError:
        GELUActivation = None
    
    for name, child in m.named_children():
        # Check for torch.nn.GELU
        if isinstance(child, nn.GELU):
            setattr(m, name, FastGELU())
            print(f"  Replaced nn.GELU at {name}")
        # Check for transformers.GELUActivation
        elif GELUActivation and isinstance(child, GELUActivation):
            setattr(m, name, FastGELU())
            print(f"  Replaced GELUActivation at {name}")
        # Also check for BERT-style activation functions that use GELU
        elif hasattr(child, 'act'):
            if isinstance(child.act, nn.GELU):
                child.act = FastGELU()
                print(f"  Replaced nn.GELU in {name}.act")
            elif GELUActivation and isinstance(child.act, GELUActivation):
                child.act = FastGELU()
                print(f"  Replaced GELUActivation in {name}.act")
        # Check if it has a hidden_act attribute that's GELU
        if hasattr(child, 'hidden_act'):
            if isinstance(child.hidden_act, nn.GELU):
                child.hidden_act = FastGELU()
                print(f"  Replaced nn.GELU in {name}.hidden_act")
            elif GELUActivation and isinstance(child.hidden_act, GELUActivation):
                child.hidden_act = FastGELU()
                print(f"  Replaced GELUActivation in {name}.hidden_act")
        
        # Recurse into child modules
        replace_gelu(child)


class ModelLoader:
    """Helper class for loading and exporting models."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            model_name: HuggingFace model name or path
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        
    def load_from_huggingface(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading {self.model_name} from HuggingFace...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model.eval()
        print("Model loaded successfully")
        
    def export_to_onnx(
        self,
        output_path: str,
        input_sample: Optional[Dict[str, torch.Tensor]] = None,
        opset_version: int = 14,
        static_shapes: bool = False
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_sample: Sample input for tracing (None for default)
            opset_version: ONNX opset version
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_from_huggingface() first.")
        
        # Create sample input if not provided
        if input_sample is None:
            input_sample = self.tokenizer(
                "This is a sample text for export",
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            )
        
        # Get input names
        input_names = list(input_sample.keys())
        output_names = ["output"]
        
        # Only set dynamic axes if we want a dynamic model
        dynamic_axes = None
        if not static_shapes:
            dynamic_axes = {name: {0: "batch_size", 1: "sequence_length"} for name in input_names}
            # Avoid guessing output dims; leave output dynamic axes unspecified unless you know them.

        print(f"Exporting to ONNX: {output_path}")
        
        # Export
        torch.onnx.export(
            self.model,
            tuple(input_sample.values()),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        # Note: ONNX Runtime will automatically optimize the graph during inference
        # For explicit optimization, you can use:
        # 1. onnxruntime.InferenceSession with graph_optimization_level
        # 2. onnxruntime-tools for model optimization
        # 3. torch.onnx.export with optimization flags (already enabled via do_constant_folding)
        
        print(f"Model exported successfully to {output_path}")
    
    def create_sample_input(
        self,
        text: str = "Sample input text",
        max_length: int = 128
    ) -> Dict[str, Any]:
        """
        Create sample input for the model.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_from_huggingface() first.")
        
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
    
    def apply_fast_gelu(self):
        replace_gelu(self.model)
