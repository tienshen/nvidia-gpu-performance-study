"""Metrics collection and analysis utilities."""

from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime
import statistics


class BenchmarkMetrics:
    """Class for collecting and analyzing benchmark metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.results: List[Dict[str, Any]] = []
        
    def add_result(self, result: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
        """
        Add a benchmark result.
        
        Args:
            result: Benchmark result dictionary
            metadata: Additional metadata (model name, config, etc.)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "metadata": metadata or {}
        }
        self.results.append(entry)
    
    def save_results(self, output_path: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def load_results(self, input_path: str) -> None:
        """
        Load results from JSON file.
        
        Args:
            input_path: Path to load results from
        """
        with open(input_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded {len(self.results)} results from {input_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No results available"}
        
        # Group by device
        by_device = {}
        for entry in self.results:
            device = entry["result"]["device"]
            if device not in by_device:
                by_device[device] = []
            by_device[device].append(entry["result"]["mean_latency_ms"])
        
        summary = {}
        for device, latencies in by_device.items():
            summary[device] = {
                "count": len(latencies),
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies)
            }
        
        return summary
    
    def compare_devices(self) -> Dict[str, Any]:
        """
        Compare performance across devices.
        
        Returns:
            Comparison metrics
        """
        summary = self.get_summary()
        
        if len(summary) < 2:
            return {"error": "Need at least 2 devices for comparison"}
        
        devices = list(summary.keys())
        comparisons = {}
        
        for i, dev1 in enumerate(devices):
            for dev2 in devices[i+1:]:
                lat1 = summary[dev1]["mean_latency_ms"]
                lat2 = summary[dev2]["mean_latency_ms"]
                speedup = lat1 / lat2 if lat2 > 0 else 0
                
                key = f"{dev1}_vs_{dev2}"
                comparisons[key] = {
                    "speedup": speedup,
                    f"{dev1}_latency_ms": lat1,
                    f"{dev2}_latency_ms": lat2
                }
        
        return comparisons
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for device, stats in summary.items():
            print(f"\n{device.upper()}:")
            print(f"  Runs: {stats['count']}")
            print(f"  Mean Latency: {stats['mean_latency_ms']:.2f} ms")
            print(f"  Median Latency: {stats['median_latency_ms']:.2f} ms")
            print(f"  Std Dev: {stats['stdev_latency_ms']:.2f} ms")
            print(f"  Min/Max: {stats['min_latency_ms']:.2f} / {stats['max_latency_ms']:.2f} ms")
        
        print("\n" + "="*60)
        
        # Print comparisons if available
        comparisons = self.compare_devices()
        if "error" not in comparisons:
            print("DEVICE COMPARISONS")
            print("="*60)
            for comp_name, comp_data in comparisons.items():
                print(f"\n{comp_name}:")
                print(f"  Speedup: {comp_data['speedup']:.2f}x")
            print("\n" + "="*60)
