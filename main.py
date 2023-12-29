import math
from torch import nn
from torch.autograd import Function
import torch

from torch.utils.cpp_extension import load
pscan = load( 'pscan', ['pscan_cuda.cpp', 'pscan_cuda_kernel.cu'], verbose=True)
torch.manual_seed(42)




def pscan_fn(A, X, Y_init):
    A = A[:, :, None].repeat(1, 1, X.size(2))
    A = A.transpose(1, 2).contiguous()
    X = X.transpose(1, 2).contiguous()
    shape = X.shape
    X = X.view(-1, shape[-1])
    A = A.view(-1, shape[-1])
    A, X = pscan.forward(A, X)
    X = X.view(shape)
    A = A.view(shape)
    return (A * Y_init[:,:, None] + X).transpose(1, 2).contiguous()

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
    for i in range(1000):
        print(i)
        Y = pscan_fn(A, X, Y_init)
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    duration = time.perf_counter() - start_time
    print(f"duration {duration}")

    s_ = Y.sum()

    print(s)
    print(s_)
