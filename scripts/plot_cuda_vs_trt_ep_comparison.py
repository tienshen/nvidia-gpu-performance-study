"""Compare CUDA EP vs TensorRT EP performance across all model configurations."""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_benchmark(path):
    """Load benchmark JSON file."""
    with open(path) as f:
        return json.load(f)

def normalize_config_key(model_name):
    """Normalize model name to a consistent configuration key for matching."""
    # Remove distilbert prefix
    name = model_name.replace('distilbert-base-uncased_', '')
    # Remove static_ prefix (TRT benchmarks don't have it)
    name = name.replace('static_', '')
    # Remove b1_s128_ or dynamic_ prefix for simplification
    name = name.replace('b1_s128_', '').replace('dynamic_', '')
    return name

def clean_label(model_name):
    """Convert model name to clean label."""
    name = model_name.replace('distilbert-base-uncased_', '')
    
    # Determine if static or dynamic
    is_static = 'b1_s128_static' in model_name or 'b1_s128_' in model_name
    is_dynamic = 'dynamic' in model_name
    
    if is_static:
        shape_type = 'Static'
        name = name.replace('b1_s128_static_', '').replace('b1_s128_', '').replace('static_', '')
    elif is_dynamic:
        shape_type = 'Dynamic'
        name = name.replace('dynamic_', '')
    else:
        shape_type = 'Static'
    
    # Replace underscores with spaces and capitalize
    parts = name.split('_')
    # Filter out 'static', 'dynamic' to avoid duplication in label
    parts = [p for p in parts if p.lower() not in ['static', 'dynamic']]
    parts = [p.upper() if p in ['fp32', 'fp16', 'int8'] else p.title() for p in parts]
    # Handle special cases
    label = ' '.join(parts)
    label = label.replace('Gelu', 'GELU').replace('Fast-Gelu', 'Fast-GELU').replace('Symmetric', '').strip()
    return f"{shape_type} {label}"

def main():
    # Load CUDA EP benchmarks
    cuda_dir = Path("results/ort-cuda/benchmarks")
    cuda_benchmarks = {}
    for json_file in cuda_dir.glob("distilbert*.json"):
        data = load_benchmark(json_file)
        config_key = normalize_config_key(data['model'])
        cuda_benchmarks[config_key] = data
    
    # Load TensorRT EP benchmarks
    trt_dir = Path("results/ort-tensorrt/benchmarks")
    trt_benchmarks = {}
    for json_file in trt_dir.glob("distilbert*.json"):
        data = load_benchmark(json_file)
        config_key = normalize_config_key(data['model'])
        trt_benchmarks[config_key] = data
    
    # Find common configurations
    common_configs = set(cuda_benchmarks.keys()) & set(trt_benchmarks.keys())
    
    if not common_configs:
        print("Error: No matching configurations found between CUDA and TensorRT benchmarks!")
        print(f"CUDA configs: {sorted(cuda_benchmarks.keys())}")
        print(f"TensorRT configs: {sorted(trt_benchmarks.keys())}")
        return
    
    # Prepare data for plotting
    configs = []
    cuda_latencies = []
    trt_latencies = []
    cuda_throughputs = []
    trt_throughputs = []
    speedups = []
    
    for config_key in sorted(common_configs):
        cuda_data = cuda_benchmarks[config_key]
        trt_data = trt_benchmarks[config_key]
        
        # Use CUDA model name for labeling (more descriptive)
        configs.append(clean_label(cuda_data['model']))
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
    
    if speedups:
        avg_speedup = np.mean(speedups)
        print(f"\nAverage TensorRT speedup: {avg_speedup:.2f}x")
        print(f"Best speedup: {max(speedups):.2f}x ({configs[speedups.index(max(speedups))]})")
        print(f"Worst speedup: {min(speedups):.2f}x ({configs[speedups.index(min(speedups))]})")
    else:
        print("\nNo speedup data available.")

if __name__ == "__main__":
    main()
