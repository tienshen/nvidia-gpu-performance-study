import subprocess

SEQ_LENS = [64, 128, 256, 384, 512]
BATCH = 1

def run_gpu_seq_sweep():
    print("Running GPU seq_len sweep...")
    for sl in SEQ_LENS:
        print(f"\n=== GPU batch {BATCH}, seq_len {sl} ===")
        subprocess.run([
            "py", "scripts/run_gpu_bench.py",
            "--batch", str(BATCH),
            "--seq-len", str(sl),
        ])

if __name__ == "__main__":
    run_gpu_seq_sweep()
