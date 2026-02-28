# Kernel Fusion: ReLU + LayerNorm

A from-scratch implementation comparing an unfused vs fused CUDA kernel for ReLU + LayerNorm on GPU using CuPy. Built to understand how kernel fusion reduces global memory traffic and improves throughput.

---

## What is Kernel Fusion

When two operations run unfused, each one launches a separate kernel. Every kernel reads from global memory, computes, and writes back to global memory. For ReLU followed by LayerNorm, that means 5 kernel launches and 8 global memory round trips.

A fused kernel does both operations in a single launch. Input is read once, ReLU is applied in registers, LayerNorm is computed using shared memory reductions, and the result is written once. 2 global memory transactions total.

The bottleneck in most GPU workloads is memory bandwidth, not compute. Fusion directly attacks that bottleneck.

---

## Operations

**ReLU**
```
f(x) = max(0, x)
```

**LayerNorm**
```
y = (x - mean(x)) / sqrt(var(x) + eps)
```

**Unfused order:** ReLU → write to global memory → LayerNorm reads back → normalize → write output

**Fused order:** read input once → ReLU in registers → mean reduction in shared memory → variance reduction in shared memory → normalize → write output once

---

## Results

Tested on RTX 3050 (4GB), batch size 4096, feature dim 1024, float32.

```
                               Unfused           Fused
Time (ms)                        1.034           0.194
Memory moved (MB)               134.22           33.55
Bandwidth (GB/s)                129.84          172.69
Speedup                          1.00x           5.32x
```

Fused kernel hits ~90% of the RTX 3050 peak memory bandwidth (192 GB/s). The speedup comes entirely from eliminating redundant global memory round trips, not from doing less compute.

---

## Project Structure

```
.
├── baseline.py       # unfused implementation using standard CuPy ops
├── fused.py          # fused RawKernel implementation in CUDA C
├── benchmark.py      # CUDA event timing and bandwidth calculation
├── plot.py           # comparison table and bar plot visualization
├── main.py           # entry point, runs benchmark and generates plot
```

---

## How it Works

**baseline.py** — runs ReLU and LayerNorm as separate CuPy operations. Each op triggers its own kernel launch and global memory read/write cycle.

**fused.py** — a single `RawKernel` written in CUDA C. Each thread block handles one row of the input. The kernel computes ReLU inline, then performs two shared memory tree reductions (mean and variance), then normalizes and writes the output in one final pass.

**benchmark.py** — uses CUDA events for accurate GPU timing. Runs 5 warmup iterations before timing. Bandwidth is calculated using theoretical memory access counts: 8 passes for unfused (5 kernel launches with multiple read/write cycles), 2 passes for fused (1 read + 1 write).

---

## Setup

```bash
pip install cupy-cuda12x matplotlib
```

Requires CUDA 12.x and a compatible NVIDIA GPU.

---

## Run

```bash
python main.py
```

Prints correctness check, terminal results table, and saves `fusion_results.png`.

---

## Key Takeaway

The fused kernel does the same math. It is faster because it touches global memory 4x less. In deep learning workloads where ops are chained — activation then normalization, normalization then projection — kernel fusion is one of the highest leverage optimizations available.