"""Script to run benchmarks on exported ONNX models."""

import argparse
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.backends import CPURunner
from bench.models import get_onnx_path, list_available_models
from bench.metrics import BenchmarkMetrics


def create_sample_input(input_metadata, batch_size=1, seq_length=128):
    """
    Create sample input based on model metadata.
    
    Args:
        input_metadata: Model input metadata
        batch_size: Batch size for input
        seq_length: Sequence length for input
        
    Returns:
        Dictionary of input tensors
    """
    inputs = {}
    
    for inp in input_metadata["inputs"]:
        name = inp["name"]
        # Common input names for transformer models
        if "input_ids" in name or "token" in name:
            # Token IDs (integers)
            inputs[name] = np.random.randint(0, 1000, size=(batch_size, seq_length), dtype=np.int64)
        elif "attention_mask" in name or "mask" in name:
            # Attention mask (all ones)
            inputs[name] = np.ones((batch_size, seq_length), dtype=np.int64)
        elif "token_type_ids" in name or "segment" in name:
            # Token type IDs (zeros for single sequence)
            inputs[name] = np.zeros((batch_size, seq_length), dtype=np.int64)
        else:
            # Generic fallback
            inputs[name] = np.random.randn(batch_size, seq_length).astype(np.float32)
    
    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on ONNX models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (without .onnx extension) or path to ONNX file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size (default: 1)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Input sequence length (default: 128)"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for CPU inference (default: None for auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: results/raw/benchmark_<timestamp>.json)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = list_available_models()
        if not models:
            print("No ONNX models found in models/ directory")
            print("Export a model first using: python scripts/export_to_onnx.py <model_name>")
        else:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        sys.exit(0)
    
    # Get model path
    if Path(args.model).exists():
        model_path = Path(args.model)
    else:
        model_path = get_onnx_path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            print("\nAvailable models:")
            for model in list_available_models():
                print(f"  - {model}")
            sys.exit(1)
    
    print(f"Running benchmark on {model_path.name}")
    print(f"Device: {args.device}")
    print(f"Iterations: {args.num_iterations} (warmup: {args.warmup_iterations})")
    print(f"Input shape: batch_size={args.batch_size}, seq_length={args.seq_length}")
    print("-" * 60)
    
    try:
        # Initialize runner based on device
        if args.device == "cpu":
            runner = CPURunner(str(model_path), num_threads=args.num_threads)
        else:  # cuda
            try:
                from bench.backends import CUDARunner
                runner = CUDARunner(str(model_path), device_id=0)
            except ImportError:
                print("Error: CUDA support not available")
                sys.exit(1)
        
        # Load model
        runner.load_model()
        
        # Get input metadata
        metadata = runner.get_input_metadata()
        print(f"\nModel inputs:")
        for inp in metadata["inputs"]:
            print(f"  {inp['name']}: {inp['shape']} ({inp['type']})")
        
        # Create sample input
        print(f"\nCreating sample input...")
        sample_input = create_sample_input(metadata, args.batch_size, args.seq_length)
        
        # Run benchmark
        print(f"\nRunning benchmark...")
        results = runner.benchmark(
            sample_input,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations
        )
        
        # Print results
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Device: {results['device']}")
        print(f"Iterations: {results['num_iterations']}")
        print(f"Mean latency: {results['mean_latency_ms']:.2f} ms")
        print(f"Min latency: {results['min_latency_ms']:.2f} ms")
        print(f"Max latency: {results['max_latency_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"Memory used: {results['memory_used_mb']:.2f} MB")
        print("="*60)
        
        # Save results
        metrics = BenchmarkMetrics()
        metrics.add_result(results, metadata={
            "model_path": str(model_path),
            "model_name": model_path.stem,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "num_threads": args.num_threads
        })
        
        if args.output:
            output_path = args.output
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/raw/benchmark_{model_path.stem}_{args.device}_{timestamp}.json"
        
        metrics.save_results(output_path)
        
        # Cleanup
        runner.cleanup()
        
    except Exception as e:
        print(f"\n✗ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
