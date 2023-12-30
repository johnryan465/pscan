import torch
import torch.nn.functional as F


torch.manual_seed(42)


from fastpscan.original import fn as original_pscan_fn
from fastpscan.cuda_v1 import fn as pscan_cuda_fn
from fastpscan.cuda_v2 import fn as pscan_cuda_v2_fn
from fastpscan.naive import fn as naive_pscan
from fastpscan.heinsen import fn as heinsen_pscan



def backward_wrapper(fn, A, X, Y_init):
    Y = fn(A, X, Y_init)
    s = Y.sum()
    gA, gX, gY_init = torch.autograd.grad(s, (A, X, Y_init), retain_graph=True)
    return gA, gX, gY_init


if __name__ == "__main__":
    import torch.utils.benchmark as benchmark


    N, T, D = 4, 1047, 3

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
    
    t3 = benchmark.Timer(
        stmt='backward_wrapper(pscan_cuda_v2_fn, A, X, Y_init)',
        setup='from __main__ import pscan_cuda_v2_fn, backward_wrapper',
        globals={'A': A, 'X': X, 'Y_init': Y_init})

    
    

    
    
    #print(tref.timeit(100))
    print(t0.timeit(1000))
    print(t1.timeit(1000))
    print(t2.timeit(1000))
    print(t3.timeit(1000))
