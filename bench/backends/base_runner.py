"""Base class for benchmark runners."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time
import psutil
import os


class BaseRunner(ABC):
    """Abstract base class for running model inference benchmarks."""
    
    def __init__(self, model_path: str, device: str):
        """
        Initialize the runner.
        
        Args:
            model_path: Path to the model file
            device: Device identifier (e.g., 'cpu', 'cuda:0')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        """
        Run inference on the given input.
        
        Args:
            input_data: Input data for the model
            
        Returns:
            Model output
        """
        pass
    
    def warmup(self, input_data: Any, num_iterations: int = 5) -> None:
        """
        Warm up the model with several inference runs.
        
        Args:
            input_data: Input data for warmup
            num_iterations: Number of warmup iterations
        """
        print(f"Warming up with {num_iterations} iterations...")
        for _ in range(num_iterations):
            self.run_inference(input_data)
    
    def benchmark(
        self,
        input_data: Any,
        num_iterations: int = 100,
        warmup_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Run benchmark and collect metrics.
        
        Args:
            input_data: Input data for benchmarking
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary containing benchmark results
        """
        # Warmup
        self.warmup(input_data, warmup_iterations)
        
        # Benchmark
        latencies = []
        process = psutil.Process(os.getpid())
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = self.run_inference(input_data)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "device": self.device,
            "num_iterations": num_iterations,
            "latencies_ms": latencies,
            "mean_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_samples_per_sec": 1000.0 / (sum(latencies) / len(latencies)),
            "memory_used_mb": memory_after - memory_before,
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
