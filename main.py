import torch
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)


from fastpscan.original import fn as original_pscan_fn
from fastpscan.cuda_v1 import fn as pscan_cuda_fn
from fastpscan.naive import fn as naive_pscan
from fastpscan.heinsen import fn as heinsen_pscan



def backward_wrapper(fn, A, X, Y_init):
    Y = fn(A, X, Y_init)
    s = Y.sum()
    gA, gX, gY_init = torch.autograd.grad(s, (A, X, Y_init), retain_graph=True)
    return gA, gX, gY_init


if __name__ != "__main__":
    import torch.utils.benchmark as benchmark


    N, T, D = 2, 1047, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_().cuda()
    X = torch.rand(N, T, D, dtype=torch.float64).requires_grad_().cuda()
    Y_init = torch.rand(N, D, dtype=torch.float64).requires_grad_().cuda()

    tref = benchmark.Timer(
        stmt='naive_pscan(A, X, Y_init)',
        setup='from __main__ import naive_pscan',
        globals={'A': A, 'X': X, 'Y_init': Y_init})

    t0 = benchmark.Timer(
        stmt='pscan_cuda_fn(A, X, Y_init)',
        setup='from __main__ import pscan_cuda_fn',
        globals={'A': A, 'X': X, 'Y_init': Y_init})

    t1 = benchmark.Timer(
        stmt='heinsen_pscan(A, X, Y_init)',
        setup='from __main__ import heinsen_pscan',
        globals={'A': A, 'X': X, 'Y_init': Y_init})
    
    t2 = benchmark.Timer(
        stmt='original_pscan_fn(A, X, Y_init)',
        setup='from __main__ import original_pscan_fn',
        globals={'A': A, 'X': X, 'Y_init': Y_init})
    
    
    #print(tref.timeit(1000))
    print(t2.timeit(1000))
    print(t1.timeit(1000))
    print(t0.timeit(1000))

if __name__ == "__main__":
    import torch.utils.benchmark as benchmark


    N, T, D = 2, 1047, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_().cuda()
    X = torch.rand(N, T, D, dtype=torch.float64).requires_grad_().cuda()
    Y_init = torch.rand(N, D, dtype=torch.float64).requires_grad_().cuda()


    t0 = benchmark.Timer(
        stmt='backward_wrapper(pscan_cuda_fn,A, X, Y_init)',
        setup='from __main__ import pscan_cuda_fn, backward_wrapper',
        globals={'A': A, 'X': X, 'Y_init': Y_init})

    t1 = benchmark.Timer(
        stmt='backward_wrapper(heinsen_pscan, A, X, Y_init)',
        setup='from __main__ import heinsen_pscan, backward_wrapper',
        globals={'A': A, 'X': X, 'Y_init': Y_init})
    
    t2 = benchmark.Timer(
        stmt='backward_wrapper(original_pscan_fn, A, X, Y_init)',
        setup='from __main__ import original_pscan_fn, backward_wrapper',
        globals={'A': A, 'X': X, 'Y_init': Y_init})
    
    
    #print(tref.timeit(100))
    print(t0.timeit(1000))
    print(t1.timeit(1000))
    print(t2.timeit(1000))



if __name__ != "__main__":
    import time, sys


    N, T, D = 2, 1047, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_().cuda()
    X = torch.rand(N, T, D, dtype=torch.float64).requires_grad_().cuda()
    Y_init = torch.rand(N, D, dtype=torch.float64).requires_grad_().cuda()

    # Iterative implementation

    y = Y_init
    s = 0

    for k in range(A.size(1)):
        y = A[:, k, None] * y + X[:, k]
        s = s + y
    Y_ = s
    s = s.sum()

    gA_ref, gX_ref, gY_init_ref = torch.autograd.grad(
        s, (A, X, Y_init), retain_graph=True
    )

    # parallel scan

    start_time = time.perf_counter()
    profile_flag = False
    if profile_flag:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(1000):
                Y = pscan_cuda_fn(A, X, Y_init)
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        for i in range(1000):
            Y = pscan_cuda_fn(A, X, Y_init)

    duration = time.perf_counter() - start_time
    print(f"duration {duration}")

    s_ = Y.sum()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        gA, gX, gY_init = torch.autograd.grad(s_, (A, X, Y_init), retain_graph=False)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("gA", gA)
    print("gA_ref", gA_ref)
    print("gA", (gA - gA_ref).norm())

    diffA = gA - gA_ref
    diffX = gX - gX_ref
    diffY_init = gY_init - gY_init_ref

    # find the top 10 largest values in diff
    #print(diffA.flatten().abs().topk(10))
    #print(diffX.flatten().abs().topk(10))
    #print(diffY_init.flatten().abs())

    #print(s)
    #print(s_)

    #print(Y)
    #print(Y_)
    #print(Y__)
    print(s)
    print(s_)
