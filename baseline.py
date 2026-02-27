# unfused cupy implementation
# pure CuPy ops, nothing custom 

import cupy as cp

def relu(x):
    return cp.maximum(0, x)

def layernorm(y, eps=1e-5):
    mean = cp.mean(y, axis=1, keepdims=True)
    var = cp.var(y, axis=1, keepdims=True)
    centered = y - mean
    normalized = centered / cp.sqrt(var + eps)
    return normalized

def relu_layernorm_unfused(x):
    y = relu(x)
    return layernorm(y)
