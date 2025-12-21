import time
import os
import numpy as np
import json
import socket
import onnxruntime as ort
from requests import session

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", f"{MODEL_NAME}.onnx")
BATCH_SIZE = 1
SEQ_LEN = 128
WARMUP = 10
RUNS = 100

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
    
    print(f"Loading ONNX model from {MODEL_PATH}")

    session = ort.InferenceSession(
        MODEL_PATH, providers=["CUDAExecutionProvider"]  # force GPU, no CPU fallback
    )
    print("Session providers:", session.get_providers())


    inputs = session.get_inputs()
    dummy_feed = {inp.name: make_dummy_input(inp) for inp in inputs}

    # Warmup
    for _ in range(WARMUP):
        session.run(None, dummy_feed)

    # Measured runs
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

    
    # Save results to json
    summary = {
        "host": socket.gethostname(),
        "model": MODEL_NAME,
        "backend": "gpu",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "runs": RUNS,
        "mean_latency_ms": float(lat.mean() * 1000),
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "throughput": float(RUNS * BATCH_SIZE / lat.sum()),
    }

    print(f"Runs: {summary['runs']}")
    print(f"Mean latency: {summary['mean_latency_ms']:.2f} ms")
    print(f"Throughput: {summary['throughput']:.2f} inferences/sec")

    os.makedirs("results/raw", exist_ok=True)
    out_path = os.path.join(
        "results", "raw",
        f"{MODEL_NAME}_gpu_bs{BATCH_SIZE}_seq{SEQ_LEN}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Hugging Face model name (used as ONNX filename)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    args = parser.parse_args()

    MODEL_NAME = args.model # override the constant
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", f"{MODEL_NAME}.onnx") # update model path
    BATCH_SIZE = args.batch  # override the constant
    SEQ_LEN = args.seq_len  # override the constant
    main()
