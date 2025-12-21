import os
import json
from glob import glob
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join("results", "raw")
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 1

def load_results():
    entries = []
    for path in glob(os.path.join(RESULT_DIR, "*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        # only care about this model + batch=1 and runs from our sweeps
        if (
            data.get("model") == MODEL_NAME
            and data.get("batch_size") == BATCH_SIZE
            and "seq_len" in data
            and "backend" in data
            and "mean_latency_ms" in data
        ):
            entries.append(data)
    return entries

def main():
    results = load_results()
    if not results:
        print("No matching results.")
        return

    # Separate CPU and GPU points
    cpu_points = {}
    gpu_points = {}

    for r in results:
        seq = r["seq_len"]
        backend = r["backend"]
        mean_ms = r["mean_latency_ms"]

        if backend == "cpu":
            cpu_points[seq] = mean_ms
        elif backend == "gpu":
            gpu_points[seq] = mean_ms

    # Sort by sequence length
    cpu_seq = sorted(cpu_points.keys())
    gpu_seq = sorted(gpu_points.keys())

    cpu_lat = [cpu_points[s] for s in cpu_seq]
    gpu_lat = [gpu_points[s] for s in gpu_seq]

    print("CPU seq scaling (seq_len -> ms):", list(zip(cpu_seq, cpu_lat)))
    print("GPU seq scaling (seq_len -> ms):", list(zip(gpu_seq, gpu_lat)))

    plt.figure()
    if cpu_seq:
        plt.plot(cpu_seq, cpu_lat, marker="o", label="CPU")
    if gpu_seq:
        plt.plot(gpu_seq, gpu_lat, marker="o", label="GPU")

    plt.xlabel("Sequence length")
    plt.ylabel("Mean latency (ms)")
    plt.title(f"{MODEL_NAME} latency vs sequence length (batch={BATCH_SIZE})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    out_path = os.path.join("results", "plots", f"{MODEL_NAME}_cpu_gpu_seq_scaling_bs{BATCH_SIZE}.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
