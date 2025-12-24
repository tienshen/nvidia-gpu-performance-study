"""Plot comparison of different DistilBERT model configurations."""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_benchmark(path):
    """Load benchmark JSON file."""
    with open(path) as f:
        return json.load(f)

def main():
    # Load all benchmark results
    benchmark_dir = Path("results/ort-cuda/benchmarks")
    
    benchmarks = []
    for json_file in benchmark_dir.glob("distilbert*.json"):
        data = load_benchmark(json_file)
        benchmarks.append(data)
    
    # Sort by latency (ascending)
    benchmarks.sort(key=lambda x: x['mean_latency_ms'])
    
    # Extract data for plotting and clean up labels
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
        label = label.replace('Gelu', 'GELU').replace('Fast-Gelu', 'Fast-GELU')
        return f"({shape_type}, {label})"
    
    models = [clean_label(b['model']) for b in benchmarks]
    latencies = [b['mean_latency_ms'] for b in benchmarks]
    throughputs = [b['throughput'] for b in benchmarks]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Latency comparison
    bars1 = ax1.bar(range(len(models)), latencies, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Model Configuration', fontsize=12)
    ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax1.set_title('CUDA EP Latency Comparison\nDistilBERT (Batch=1, SeqLen=128)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, latencies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}ms',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Throughput comparison
    bars2 = ax2.bar(range(len(models)), throughputs, color='coral', alpha=0.7)
    ax2.set_xlabel('Model Configuration', fontsize=12)
    ax2.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax2.set_title('CUDA EP Throughput Comparison\nDistilBERT (Batch=1, SeqLen=128)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, throughputs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("results/ort-cuda/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "distilbert_config_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CUDA EP Performance Summary (sorted by latency)")
    print("="*60)
    for b in benchmarks:
        model_name = b['model'].replace('distilbert-base-uncased_', '')
        print(f"{model_name:30s}: {b['mean_latency_ms']:6.2f}ms  {b['throughput']:7.2f} samples/sec")
    print("="*60)

if __name__ == "__main__":
    main()
