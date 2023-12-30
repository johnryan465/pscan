import torch

torch.manual_seed(42)


from fastpscan.original import fn as original_pscan_fn
from fastpscan.cuda_v1 import fn as pscan_cuda_fn
from fastpscan.naive import fn as naive_pscan
from fastpscan.heinsen import fn as heinsen_pscan


if __name__ == "__main__":
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
