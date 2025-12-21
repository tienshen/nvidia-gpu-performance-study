import os
import json
from glob import glob
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join("results", "raw")
MODEL_NAME = "bert-base-uncased"
SEQ_LEN = 128
BATCH_SIZE = 1  # focus on batch=1 for now


def load_results():
    entries = []
    for path in glob(os.path.join(RESULT_DIR, "*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        if (
            data.get("model") == MODEL_NAME
            and data.get("seq_len") == SEQ_LEN
            and data.get("batch_size") == BATCH_SIZE
        ):
            entries.append(data)
    return entries


def main():
    results = load_results()
    if not results:
        print("No matching results found.")
        return

    print(f"Results for model={MODEL_NAME}, seq_len={SEQ_LEN}, batch={BATCH_SIZE}")
    for r in results:
        print(
            f"- host={r.get('host','?'):15s} "
            f"backend={r['backend']:3s} "
            f"mean={r['mean_latency_ms']:.2f} ms "
            f"p50={r.get('p50_ms', 0):.2f} ms "
            f"p90={r.get('p90_ms', 0):.2f} ms "
            f"p99={r.get('p99_ms', 0):.2f} ms "
            f"throughput={r['throughput']:.2f} inf/s"
        )


    # Filter for this host only (PC), separate cpu/gpu
    hosts = {r["host"] for r in results}
    for host in hosts:
        host_res = [r for r in results if r["host"] == host]
        if len(host_res) < 2:
            continue
        backends = [r["backend"] for r in host_res]
        thr = [r["throughput"] for r in host_res]

        plt.figure()
        plt.bar(backends, thr)
        plt.xlabel("Backend")
        plt.ylabel("Throughput (inferences/sec)")
        plt.title(f"{MODEL_NAME} throughput (batch={BATCH_SIZE}) on {host}")
        os.makedirs(os.path.join("results", "plots"), exist_ok=True)
        out_path = os.path.join(
            "results", "plots",
            f"{MODEL_NAME}_{host}_cpu_vs_gpu_bs{BATCH_SIZE}.png"
        )
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
