import os
import json
from glob import glob
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join("results", "raw")
MODEL_NAME = "bert-base-uncased"
SEQ_LEN = 128
BATCHES = [1, 2, 4, 8, 16, 32]

def load_results():
    entries = []
    for path in glob(os.path.join(RESULT_DIR, "*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        if (
            data.get("model") == MODEL_NAME
            and data.get("seq_len") == SEQ_LEN
            and data.get("batch_size") in BATCHES
        ):
            entries.append(data)
    return entries

def main():
    results = load_results()
    if not results:
        print("No matching results.")
        return

    cpu_thr = []
    gpu_thr = []
    x = [str(b) for b in BATCHES]

    for b in BATCHES:
        cpu = next((r for r in results if r["backend"] == "cpu" and r["batch_size"] == b), None)
        gpu = next((r for r in results if r["backend"] == "gpu" and r["batch_size"] == b), None)
        cpu_thr.append(cpu["throughput"] if cpu else 0.0)
        gpu_thr.append(gpu["throughput"] if gpu else 0.0)

    positions = range(len(BATCHES))
    width = 0.35

    plt.figure()
    plt.bar([p - width/2 for p in positions], cpu_thr, width, label="CPU")
    plt.bar([p + width/2 for p in positions], gpu_thr, width, label="GPU")

    plt.xticks(list(positions), x)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (inferences/sec)")
    plt.title(f"{MODEL_NAME} throughput vs batch size (CPU vs GPU)")
    plt.legend()

    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    out_path = os.path.join("results", "plots", f"{MODEL_NAME}_cpu_gpu_throughput_batch_sweep.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
