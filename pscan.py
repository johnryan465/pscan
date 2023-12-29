import math
from torch import nn
from torch.autograd import Function
import torch

import pscan

torch.manual_seed(42)



pscan = pscan_cuda.forward

if __name__ == "__main__":
    import time, sys

    N, T, D = 2, 1047, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_().cuda()
    X = torch.randn(N, T, D, dtype=torch.float64).requires_grad_().cuda()
    Y_init = torch.randn(N, D, dtype=torch.float64).requires_grad_().cuda()

    # Iterative implementation

    y = Y_init
    s = 0

    for k in range(A.size(1)):
        y = A[:, k, None] * y + X[:, k]
        s = s + y

    s = s.sum()

    gA_ref, gX_ref, gY_init_ref = torch.autograd.grad(
        s, (A, X, Y_init), retain_graph=True
    )

    # parallel scan

    start_time = time.perf_counter()
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #for i in range(1000):
    #    print(i)
    Y = pscan(A, X, Y_init)
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    duration = time.perf_counter() - start_time
    print(f"duration {duration}")

    s_ = Y.sum()

    print(s)
    print(s_)
