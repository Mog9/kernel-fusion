#warmup / timing / compare

import cupy as cp

def benchmark(fn, x, iter = 20):
    for _ in range(5):
        fn(x)
    cp.cuda.runtime.deviceSynchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iter):
        fn(x)
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    return elapsed_ms / iter


def compute_bandwidth(B, N, time_ms, passes, dtype_bytes=4):
    elements = B * N
    bytes_per_tensor = elements * dtype_bytes
    total_bytes = bytes_per_tensor * passes
    time_s = time_ms / 1000
    bandwidth_gbs = total_bytes / time_s / 1e9

    return total_bytes / 1e6, bandwidth_gbs #mb/gb
