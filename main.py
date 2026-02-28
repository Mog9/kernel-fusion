import cupy as cp
from baseline import relu_layernorm_unfused
from fused import relu_layernorm_fused
from benchmark import benchmark, compute_bandwidth_unfused, compute_bandwidth_fused
from plot import plot


def collect_results():
    B, N = 4096, 1024
    x = cp.random.randn(B, N, dtype=cp.float32)

    out_unfused = relu_layernorm_unfused(x)
    out_fused = relu_layernorm_fused(x)
    max_diff = float(cp.max(cp.abs(out_unfused - out_fused)))
    print(f"correctness check — max diff: {max_diff:.6f}")

    time_unfused = benchmark(relu_layernorm_unfused, x)
    mem_unfused, bw_unfused = compute_bandwidth_unfused(B, N, time_unfused)

    time_fused = benchmark(relu_layernorm_fused, x)
    mem_fused, bw_fused = compute_bandwidth_fused(B, N, time_fused)

    speedup = time_unfused / time_fused

    print(f"\n{'':30s} {'Unfused':>15} {'Fused':>15}")
    print(f"{'-'*60}")
    print(f"{'Time (ms)':30s} {time_unfused:>15.3f} {time_fused:>15.3f}")
    print(f"{'Memory moved (MB)':30s} {mem_unfused:>15.2f} {mem_fused:>15.2f}")
    print(f"{'Bandwidth (GB/s)':30s} {bw_unfused:>15.2f} {bw_fused:>15.2f}")
    print(f"{'Speedup':30s} {'1.00x':>15} {speedup:>14.2f}x")

    return {
        "time_unfused": time_unfused,
        "time_fused": time_fused,
        "mem_unfused": mem_unfused,
        "mem_fused": mem_fused,
        "bw_unfused": bw_unfused,
        "bw_fused": bw_fused,
        "speedup": speedup,
    }


if __name__ == "__main__":
    results = collect_results()
    plot(results)