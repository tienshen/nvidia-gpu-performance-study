"""Compare CUDA EP vs TensorRT EP performance across all model configurations."""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_benchmark(path):
    """Load benchmark JSON file."""
    with open(path) as f:
        return json.load(f)

def clean_label(model_name):
    """Convert model name to clean label."""
    name = model_name.replace('distilbert-base-uncased_', '')
    
    # Determine if static or dynamic
    is_static = name.startswith('b1_s128_')
    shape_type = 'Static' if is_static else 'Dynamic'
    
    # Remove b1_s128_ prefix if present
    name = name.replace('b1_s128_', '')
    
    # Replace underscores with spaces and capitalize
    parts = name.split('_')
    parts = [p.upper() if p in ['fp32', 'fp16', 'int8'] else p.title() for p in parts]
    # Handle special cases
    label = ', '.join(parts)
    label = label.replace('Gelu', 'GELU').replace('Fast-Gelu', 'Fast-GELU').replace('Symmetric', '')
    return f"{shape_type} {label}"

def main():
    # Load CUDA EP benchmarks
    cuda_dir = Path("results/ort-cuda/benchmarks")
    cuda_benchmarks = {}
    for json_file in cuda_dir.glob("distilbert*.json"):
        data = load_benchmark(json_file)
        cuda_benchmarks[data['model']] = data
    
    # Load TensorRT EP benchmarks
    trt_dir = Path("results/ort-tensorrt/benchmarks")
    trt_benchmarks = {}
    for json_file in trt_dir.glob("distilbert*.json"):
        data = load_benchmark(json_file)
        trt_benchmarks[data['model']] = data
    
    # Find common models
    common_models = set(cuda_benchmarks.keys()) & set(trt_benchmarks.keys())
    
    # Prepare data for plotting
    configs = []
    cuda_latencies = []
    trt_latencies = []
    cuda_throughputs = []
    trt_throughputs = []
    speedups = []
    
    for model in sorted(common_models):
        cuda_data = cuda_benchmarks[model]
        trt_data = trt_benchmarks[model]
        
        configs.append(clean_label(model))
        cuda_latencies.append(cuda_data['mean_latency_ms'])
        trt_latencies.append(trt_data['mean_latency_ms'])
        cuda_throughputs.append(cuda_data['throughput'])
        trt_throughputs.append(trt_data['throughput'])
        speedups.append(cuda_data['mean_latency_ms'] / trt_data['mean_latency_ms'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Latency comparison
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cuda_latencies, width, label='CUDA EP', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, trt_latencies, width, label='TensorRT EP', color='darkgreen', alpha=0.8)
    
    ax1.set_xlabel('Model Configuration', fontsize=12)
    ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax1.set_title('CUDA EP vs TensorRT EP Latency\nDistilBERT (Batch=1, SeqLen=128)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Throughput comparison
    x2 = np.arange(len(configs))
    
    bars3 = ax2.bar(x2 - width/2, cuda_throughputs, width, label='CUDA EP', color='steelblue', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, trt_throughputs, width, label='TensorRT EP', color='darkgreen', alpha=0.8)
    
    ax2.set_xlabel('Model Configuration', fontsize=12)
    ax2.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax2.set_title('CUDA EP vs TensorRT EP Throughput\nDistilBERT (Batch=1, SeqLen=128)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("results/cuda-vs-trt-ep")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cuda_vs_tensorrt_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*100)
    print("CUDA EP vs TensorRT EP Performance Comparison")
    print("="*100)
    print(f"{'Configuration':<40} {'CUDA (ms)':<12} {'TRT (ms)':<12} {'Speedup':<12} {'CUDA (samp/s)':<15} {'TRT (samp/s)':<15}")
    print("-"*100)
    for i, config in enumerate(configs):
        print(f"{config:<40} {cuda_latencies[i]:>10.2f}   {trt_latencies[i]:>10.2f}   {speedups[i]:>8.2f}x   {cuda_throughputs[i]:>13.1f}   {trt_throughputs[i]:>13.1f}")
    print("="*100)
    
    avg_speedup = np.mean(speedups)
    print(f"\nAverage TensorRT speedup: {avg_speedup:.2f}x")
    print(f"Best speedup: {max(speedups):.2f}x ({configs[speedups.index(max(speedups))]})")
    print(f"Worst speedup: {min(speedups):.2f}x ({configs[speedups.index(min(speedups))]})")

if __name__ == "__main__":
    main()
