import subprocess

SEQ_LENS = [64, 128, 256, 384, 512]
BATCH = 1

def run_cpu_seq_sweep():
    print("Running CPU seq_len sweep...")
    for sl in SEQ_LENS:
        print(f"\n=== CPU batch {BATCH}, seq_len {sl} ===")
        subprocess.run([
            "py", "scripts/run_cpu_bench.py",
            "--batch", str(BATCH),
            "--seq-len", str(sl),
        ])

if __name__ == "__main__":
    run_cpu_seq_sweep()
