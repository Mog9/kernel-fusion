#warmup / timing / compare

import cupy as cp

def benchmark(fn, x, iter = 20):
    for _ in range(5): #warmup
        fn(x)
    cp.cuda.runtime.deviceSynchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iter):
        fn(x)
    end.record()
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / iter


def compute_bandwidth_unfused(B, N, time_ms, dtype_bytes=4):
    passes = 8
    total_bytes = B * N * dtype_bytes * passes
    bandwidth_gbs = total_bytes / (time_ms / 1000) / 1e9
    return total_bytes / 1e6, bandwidth_gbs


def compute_bandwidth_fused(B, N, time_ms, dtype_bytes=4):
    passes = 2
    total_bytes = B * N * dtype_bytes * passes
    bandwidth_gbs = total_bytes / (time_ms / 1000) / 1e9
    return total_bytes / 1e6, bandwidth_gbs
