"""CPU-based inference runner using ONNX Runtime."""

import numpy as np
import onnxruntime as ort
from typing import Any, Dict
from .base_runner import BaseRunner


class CPURunner(BaseRunner):
    """Runner for CPU-based inference using ONNX Runtime."""
    
    def __init__(self, model_path: str, num_threads: int = None):
        """
        Initialize CPU runner.
        
        Args:
            model_path: Path to ONNX model file
            num_threads: Number of threads for inference (None for default)
        """
        super().__init__(model_path, "cpu")
        self.num_threads = num_threads
        self.session = None
        
    def load_model(self) -> None:
        """Load ONNX model for CPU inference."""
        sess_options = ort.SessionOptions()
        
        if self.num_threads is not None:
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = self.num_threads
        
        # Use CPU execution provider
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"Model loaded on CPU")
        if self.num_threads:
            print(f"Using {self.num_threads} threads")
        
    def run_inference(self, input_data: Dict[str, np.ndarray]) -> Any:
        """
        Run inference on CPU.
        
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
