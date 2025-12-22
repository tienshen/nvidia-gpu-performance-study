"""CUDA-based inference runner using ONNX Runtime."""

import numpy as np
import onnxruntime as ort
from typing import Any, Dict
from .base_runner import BaseRunner


class CUDARunner(BaseRunner):
    """Runner for CUDA-based inference using ONNX Runtime."""
    
    def __init__(self, model_path: str, device_id: int = 0):
        """
        Initialize CUDA runner.
        
        Args:
            model_path: Path to ONNX model file
            device_id: CUDA device ID
        """
        super().__init__(model_path, f"cuda:{device_id}")
        self.device_id = device_id
        self.session = None
        
    def load_model(self) -> None:
        """Load ONNX model for CUDA inference."""
        sess_options = ort.SessionOptions()
        
        # Use CUDA execution provider
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': self.device_id,
            }),
            'CPUExecutionProvider'  # Fallback
        ]
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"Model loaded on CUDA device {self.device_id}")
        print(f"Available providers: {self.session.get_providers()}")
        
    def run_inference(self, input_data: Dict[str, np.ndarray]) -> Any:
        """
        Run inference on CUDA.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            
        Returns:
            Model outputs
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        outputs = self.session.run(None, input_data)
        return outputs
    
    def get_input_metadata(self) -> Dict[str, Any]:
        """Get model input metadata."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            "inputs": [
                {
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type
                }
                for inp in self.session.get_inputs()
            ]
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.session is not None:
            del self.session
            self.session = None
