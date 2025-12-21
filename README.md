# ⚡️ NVIDIA GPU Inference Diagnosis

### Systematically comparing ONNX Runtime CUDA, TensorRT EP, and native TensorRT builds

---

## 1. Motivation

The CoreML investigation surfaced how front-end graph structure and runtime scheduling dictate real-world performance. I want the same level of visibility for **NVIDIA GPUs**:

- When does ONNX Runtime with **CUDAExecutionProvider** saturate the GPU?
- What extra benefit (or complexity) does the **TensorRT Execution Provider** add?
- When is it worth exporting to **native TensorRT engines** instead of staying in ORT?

This repository is the staging ground for that diagnosis.

---

## 2. Current Assets

| Asset | Description |
| --- | --- |
| `results/plots/bert-base-uncased_cpu_gpu_seq_scaling_bs1.png` | Baseline latency vs sequence length (CPU vs RTX 3060 Ti) |
| `results/plots/bert-base-uncased_cpu_gpu_throughput_batch_sweep.png` | Throughput vs batch size (CPU vs RTX 3060 Ti) |
| `scripts/compare_cpu_gpu.py` | Convenience runner to compare CPU vs GPU providers |
| `scripts/run_gpu_bench.py` | Core benchmarking loop for GPU runs |
| `scripts/run_seq_sweep_gpu.py` | Sequence-length scaling harness |
| `scripts/run_cpu_bench.py`, `run_seq_sweep_cpu.py` | CPU control experiments |
| `scripts/export_to_onnx.py` | Export helpers for BERT/DistilBERT/tiny-systems-bert |
| `scripts/plot_seq_scaling_cpu_gpu.py`, `plot_throughput_batch_sweep.py` | Plotters that generated the seeded figures |

Everything currently references the **RTX 3060 Ti + Ryzen 3700X** lab box used in the original transformer study.

---

## 3. Planned Scope

### Phase 1 — ORT + CUDA EP: establish ground truth

Purpose: Identify real bottlenecks before introducing aggressive compiler/runtime transformations.

What to emphasize:

Fixed input regimes (batch / sequence length)

Nsight Systems + Compute used to answer:

where time is actually spent,

how much is kernel launch vs math,

where CPU↔GPU synchronization appears.

Findings framed as constraints (not optimizations yet).

Example wording:

“Used ORT + CUDA EP to establish a reproducible performance baseline and identify dominant execution bottlenecks under controlled Transformer input regimes.”

### Phase 2 — ORT + TensorRT EP: controlled acceleration & attribution

Purpose: Introduce TensorRT selectively while preserving framework-level structure.

What to emphasize:

Partial graph conversion

Identification of:

which subgraphs convert,

where dynamic shapes prevent fusion,

where fallbacks occur.

Insight into why acceleration is limited.

Example wording:

“Integrated TensorRT via ORT’s execution provider to study subgraph conversion, fusion boundaries, and shape-driven fragmentation without abandoning the ONNX runtime.”

This sounds extremely intentional.

### Phase 3 — Native TensorRT: manual optimization ceiling

Purpose: Remove abstraction limits and push maximum performance for known regimes.

What to emphasize:

Explicit optimization profiles

Precision control (FP16 / INT8 if applicable)

Engine-level profiling correlated with Nsight

Tradeoffs:

latency vs throughput,

flexibility vs determinism.

Example wording:

“Transitioned to native TensorRT to manually define optimization profiles and precision strategies, enabling deeper fusion and higher throughput once input distributions were well-characterized.”

Each phase will mirror the documentation style of the CoreML project: qualitative summary in the README, quantitative details in appendices plus `results/txt/` profile dumps.

---

## 4. Repository Layout

```
nvidia-gpu-performance-study/
├── README.md
├── scripts/                      # benchmarking + plotting harnesses (seeded from CoreML study)
│   ├── compare_cpu_gpu.py
│   ├── export_to_onnx.py
│   ├── plot_seq_scaling_cpu_gpu.py
│   ├── plot_throughput_batch_sweep.py
│   ├── run_benchmarks.py
│   ├── run_cpu_bench.py
│   ├── run_gpu_bench.py
│   ├── run_seq_sweep_cpu.py
│   └── run_seq_sweep_gpu.py
├── results/
│   ├── plots/                    # seeded PNGs + future figures
│   └── csv/                      # placeholder for raw measurement tables
└── notebooks/                    # optional exploratory analysis
```

---

## 5. Next Steps

- [ ] Wire up ORT profiling for CUDA EP runs (`run_gpu_bench.py --profile-dir ...`)  
- [ ] Add Nsight Systems capture script to `scripts/`  
- [ ] Draft Phase 1 findings (CPU vs CUDA EP) in README section 5  
- [ ] Start collecting TensorRT EP data for Phase 2  
- [ ] Extend `results/txt/` with profiler summaries (mirroring CoreML appendices)

Once Phase 1 data is in place, I’ll mirror the CoreML report structure (Key Observations → Appendices) so both projects can be compared side by side.
