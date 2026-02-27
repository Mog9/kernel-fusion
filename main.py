import cupy as cp
from benchmark import benchmark, compute_bandwidth
from baseline import relu_layernorm_unfused

B, N = 4096, 1024
x = cp.random.randn(B, N, dtype=cp.float32)

time_unfused = benchmark(relu_layernorm_unfused, x)
estimated_passes = 5
mem_mb, bandwidth = compute_bandwidth(B, N, time_unfused,estimated_passes)

print(f"Unfused time: {time_unfused:.3f} ms")
print(f"Estimated memory moved: {mem_mb:.2f} MB")
print(f"Effective bandwidth: {bandwidth:.2f} GB/s")
