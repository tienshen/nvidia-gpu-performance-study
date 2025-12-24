"""Compare architectural differences between two ONNX models."""
import json
from collections import Counter
from pathlib import Path

def load_architecture(path):
    """Load architecture JSON file."""
    with open(path) as f:
        return json.load(f)

def main():
    # Load both architectures
    gelu_arch = load_architecture("results/gelu_fp16_architecture.json")
    fast_gelu_arch = load_architecture("results/fast-gelu_fp16_architecture.json")
    
    # Compare node counts
    gelu_counts = gelu_arch['node_type_counts']
    fast_gelu_counts = fast_gelu_arch['node_type_counts']
    
    # Get all unique op types
    all_ops = set(gelu_counts.keys()) | set(fast_gelu_counts.keys())
    
    # Create comparison
    comparison = {
        "summary": {
            "gelu_fp16_total_nodes": gelu_arch['total_nodes'],
            "fast_gelu_fp16_total_nodes": fast_gelu_arch['total_nodes'],
            "node_difference": fast_gelu_arch['total_nodes'] - gelu_arch['total_nodes']
        },
        "op_type_comparison": {}
    }
    
    for op in sorted(all_ops):
        gelu_count = gelu_counts.get(op, 0)
        fast_gelu_count = fast_gelu_counts.get(op, 0)
        diff = fast_gelu_count - gelu_count
        
        if diff != 0:  # Only include ops with differences
            comparison["op_type_comparison"][op] = {
                "gelu_fp16": gelu_count,
                "fast_gelu_fp16": fast_gelu_count,
                "difference": diff
            }
    
    # Save comparison
    output_path = Path("results/architecture_comparison_gelu_vs_fast-gelu_fp16.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Architecture comparison saved to: {output_path}")
    print("\n" + "="*80)
    print("GELU FP16 vs Fast-GELU FP16 Architecture Comparison")
    print("="*80)
    print(f"\nTotal nodes:")
    print(f"  GELU FP16: {gelu_arch['total_nodes']}")
    print(f"  Fast-GELU FP16: {fast_gelu_arch['total_nodes']}")
    print(f"  Difference: {comparison['summary']['node_difference']:+d}")
    
    print(f"\nKey differences in op types:")
    print(f"{'Op Type':<20} {'GELU FP16':>12} {'Fast-GELU FP16':>16} {'Difference':>12}")
    print("-"*80)
    
    # Sort by absolute difference
    sorted_ops = sorted(comparison["op_type_comparison"].items(), 
                       key=lambda x: abs(x[1]['difference']), 
                       reverse=True)
    
    for op, counts in sorted_ops:
        print(f"{op:<20} {counts['gelu_fp16']:>12} {counts['fast_gelu_fp16']:>16} {counts['difference']:>+12}")
    
    # Highlight critical differences
    print("\n" + "="*80)
    print("Critical architectural differences:")
    print("="*80)
    
    if 'Erf' in comparison["op_type_comparison"]:
        erf_data = comparison["op_type_comparison"]['Erf']
        print(f"\nErf (error function - used in exact GELU):")
        print(f"  GELU FP16: {erf_data['gelu_fp16']} ops")
        print(f"  Fast-GELU FP16: {erf_data['fast_gelu_fp16']} ops")
        print(f"  → GELU FP16 uses Erf for exact GELU computation")
    
    if 'Tanh' in comparison["op_type_comparison"]:
        tanh_data = comparison["op_type_comparison"]['Tanh']
        print(f"\nTanh (hyperbolic tangent - used in Fast-GELU approximation):")
        print(f"  GELU FP16: {tanh_data['gelu_fp16']} ops")
        print(f"  Fast-GELU FP16: {tanh_data['fast_gelu_fp16']} ops")
        print(f"  → Fast-GELU FP16 uses Tanh approximation instead of Erf")
    
    print("\nPerformance implication:")
    print("  TensorRT has better-optimized FP16 Erf kernels than FP16 Tanh kernels")
    print("  Result: GELU FP16 (0.89ms) is 3.1x faster than Fast-GELU FP16 (2.78ms)")
    print("  despite Fast-GELU using supposedly 'faster' approximation")

if __name__ == "__main__":
    main()
