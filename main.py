import torch
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)


from fastpscan import original
original_pscan_fn = original.PScan.apply

from fastpscan.cuda_v1 import FastPScan


def pscan_fn_(A, X):
    A = A[:, :, None].repeat(1, 1, X.size(2)).clone()
    A = A.transpose(1, 2).contiguous()
    X = X.transpose(1, 2).contiguous()
    shape = X.shape
    X = X.view(-1, shape[-1])
    A = A.view(-1, shape[-1])
    C = torch.stack([A, X], dim=2).contiguous()
    C = pscan.forward(C)
    A_ = C[:,:,0].view(shape).transpose(1, 2)
    X_ = C[:,:,1].view(shape).transpose(1, 2)
    return A_, X_

def default_pscan(A, X, Y_init):
    y = Y_init
    s = 0

    for k in range(A.size(1)):
        y = A[:, k, None] * y + X[:, k]
        s = s + y
    Y_ = s
    return Y_

pscan_cuda_fn = FastPScan.apply

def fast_pscan(A, X, Y_init):
    Y_init = Y_init[:, :, None]
    Xa = torch.concat([Y_init, torch.transpose(X, 1, 2)], dim=-1)
    X_real = torch.abs(Xa).log()
    X_complex = (Xa < 0).to(torch.float64)
    A_real = torch.abs(A).log()
    X_ = torch.complex(X_real, X_complex * torch.pi)
    A_complex = (A < 0).to(torch.float64)
    A_ = torch.complex(A_real, A_complex * torch.pi)
    a_star =  F.pad(torch.cumsum(A_, dim=-1), (1,0))[:,None,:] 
    log_x0_plus_b_star = torch.logcumsumexp(X_ - a_star, dim=-1)
    log_x =  a_star + log_x0_plus_b_star
    return torch.transpose(torch.exp(log_x).real[:,:,1:], 1, 2)


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
        stmt='default_pscan(A, X, Y_init)',
        setup='from __main__ import default_pscan',
        globals={'A': A, 'X': X, 'Y_init': Y_init})

    t0 = benchmark.Timer(
        stmt='pscan_cuda_fn(A, X, Y_init)',
        setup='from __main__ import pscan_cuda_fn',
        globals={'A': A, 'X': X, 'Y_init': Y_init})

    t1 = benchmark.Timer(
        stmt='fast_pscan(A, X, Y_init)',
        setup='from __main__ import fast_pscan',
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
        stmt='backward_wrapper(fast_pscan, A, X, Y_init)',
        setup='from __main__ import fast_pscan, backward_wrapper',
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
