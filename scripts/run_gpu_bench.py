import time
import os
import numpy as np
import json
import socket
import onnxruntime as ort
from pathlib import Path

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", f"{MODEL_NAME}.onnx")
BATCH_SIZE = 1
SEQ_LEN = 128
WARMUP = 50
RUNS = 250
ENABLE_PROFILING = False
PROFILE_DIR = None
EXECUTION_PROVIDER = "cuda"

def make_dummy_input(input_def):
    shape = [BATCH_SIZE, SEQ_LEN]
    # match CPU version (simplified)
    name = input_def.name.lower()
    if "input_ids" in name:
        return np.random.randint(0, 30000, size=shape, dtype=np.int64)
    elif "token_type" in name:
        return np.zeros(shape, dtype=np.int64)
    elif "attention" in name:
        return np.ones(shape, dtype=np.int64)
    else:
        return np.random.randint(0, 10000, size=shape, dtype=np.int64)

def main():
    if EXECUTION_PROVIDER not in ["cuda", "tensorrt"]:
        raise ValueError(f"Unsupported execution provider: {EXECUTION_PROVIDER}. Only 'cuda' and 'tensorrt' are supported.")
    
    print(f"Loading ONNX model from {MODEL_PATH}")

    sess_options = ort.SessionOptions()
    
    # Enable profiling if requested
    profile_output = None
    if ENABLE_PROFILING:
        profile_dir = Path(PROFILE_DIR)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_output = profile_dir / f"{MODEL_NAME}_b{BATCH_SIZE}_s{SEQ_LEN}_profile.json"
        
        # Remove existing profile if it exists
        if profile_output.exists():
            profile_output.unlink()
        
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = str(profile_output.with_suffix(''))
        print(f"Profiling enabled: {profile_output}")
    
    # Configure execution provider
    if EXECUTION_PROVIDER == "tensorrt":
        trt_provider_options = {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,  # 2GB
            'trt_fp16_enable': False,
            'trt_int8_enable': False,
        }
        providers = [
            ('TensorrtExecutionProvider', trt_provider_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
    else:  # cuda
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=providers
    )
    print("Session providers:", session.get_providers())


    inputs = session.get_inputs()
    dummy_feed = {inp.name: make_dummy_input(inp) for inp in inputs}

    # Warmup
    print(f"Warming up ({WARMUP} runs)...")
    for _ in range(WARMUP):
        session.run(None, dummy_feed)

    # Measured runs
    print(f"Running benchmark ({RUNS} runs)...")
    latencies = []
    for _ in range(RUNS):
        start = time.perf_counter()
        session.run(None, dummy_feed)
        end = time.perf_counter()
        latencies.append(end - start)

    lat = np.array(latencies)

    # Obtain percentile latencies
    p50 = float(np.percentile(lat, 50) * 1000)
    p90 = float(np.percentile(lat, 90) * 1000)
    p99 = float(np.percentile(lat, 99) * 1000)

    # Determine backend label and output directory based on EP
    backend_label = "tensorrt" if EXECUTION_PROVIDER == "tensorrt" else "gpu"
    output_dir = "ort-tensorrt" if EXECUTION_PROVIDER == "tensorrt" else "ort-cuda"
    filename_suffix = "trt" if EXECUTION_PROVIDER == "tensorrt" else "gpu"
    
    # Save results to json
    summary = {
        "host": socket.gethostname(),
        "model": MODEL_NAME,
        "backend": backend_label,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "runs": RUNS,
        "mean_latency_ms": float(lat.mean() * 1000),
        "std_latency_ms": float(lat.std() * 1000),
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "throughput": float(RUNS * BATCH_SIZE / lat.sum()),
    }

    print(f"\nRuns: {summary['runs']}")
    print(f"Mean latency: {summary['mean_latency_ms']:.2f} ms")
    print(f"Std latency: {summary['std_latency_ms']:.2f} ms")
    print(f"P50: {p50:.2f} ms")
    print(f"P90: {p90:.2f} ms")
    print(f"P99: {p99:.2f} ms")
    print(f"Throughput: {summary['throughput']:.2f} inferences/sec")

    # End profiling and rename to remove timestamp
    if ENABLE_PROFILING:
        profile_file_with_timestamp = session.end_profiling()
        # Rename to remove timestamp
        import shutil
        shutil.move(profile_file_with_timestamp, str(profile_output))
        print(f"Profile saved to: {profile_output}")

    os.makedirs(f"results/{output_dir}/benchmarks", exist_ok=True)
    out_path = os.path.join(
        "results", output_dir, "benchmarks",
        f"{MODEL_NAME}_{filename_suffix}_bs{BATCH_SIZE}_seq{SEQ_LEN}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model name (used as ONNX filename)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--ep", type=str, choices=["cuda", "tensorrt"], default="cuda",
                        help="Execution provider: 'cuda' or 'tensorrt' (default: cuda)")
    parser.add_argument("--profile", action="store_true", help="Enable ONNX Runtime profiling")
    parser.add_argument("--profile-dir", type=str, default=None,
                        help="Directory to save profile data (default: auto based on EP)")
    args = parser.parse_args()

    MODEL_NAME = args.model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", f"{MODEL_NAME}.onnx")
    BATCH_SIZE = args.batch
    SEQ_LEN = args.seq_len
    EXECUTION_PROVIDER = args.ep
    ENABLE_PROFILING = args.profile
    
    # Set default profile directory based on EP if not specified
    if args.profile_dir:
        PROFILE_DIR = args.profile_dir
    else:
        # Use consistent naming: ort-tensorrt for tensorrt, ort-cuda for cuda
        ep_dir = "ort-tensorrt" if args.ep == "tensorrt" else "ort-cuda"
        PROFILE_DIR = f"results/{ep_dir}/profiles"
    
    main()
