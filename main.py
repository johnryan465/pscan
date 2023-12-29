import math
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity


from torch.utils.cpp_extension import load
pscan = load( 'pscan', ['pscan_cuda.cpp', 'pscan_cuda_kernel.cu'], verbose=True)
torch.manual_seed(42)


class FastPScan(torch.autograd.Function):
    # Given A is NxTx1 and X is NxTxD, expands A and X in place in O(T),
    # and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]
    @staticmethod
    def pscan_fn(A, X):
        A = A.repeat(1, 1, X.size(2)) # .clone()
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

    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        FastPScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star, ctx.X_star = FastPScan.pscan_fn(ctx.A, X)
        return ctx.A_star * ctx.Y_init + ctx.X_star
        # return ctx.res

    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        A_ = torch.flip(A, [1])
        #print("A", A)
        #print("R", R)
        A__ = torch.cat([torch.ones_like(A_[:, -1:]), A_[:, :-1]], dim=1)
        rev_r = torch.flip(R, [1])
        #print("A__", A__)
        #print("rev_r", rev_r)
        _, R_ = FastPScan.pscan_fn(A__, rev_r)
        R = torch.flip(R_, [1])
        #FastPScan.acc_rev_(A, R)
        #print("R__test", R__test)
        #print("R", R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)

class PScan(torch.autograd.Function):
    # Given A is NxTx1 and X is NxTxD, expands A and X in place in O(T),
    # and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        """
        Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
        A'[i, t] = A[i, 2*t] * A[i, 2*t - 1]
        X'[i, t] = A[i, 2*t] * X[i, 2*t - 1] + X[i, 2*t]

        Build a Fenwick Tree
        """
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))
            A[:, -1].mul_(A[:, -2])


    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        # ppprint(grad_output)
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)
    
original_pscan_fn = PScan.apply

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
        stmt='original_pscan_fn(A, X, Y_init)',
        setup='from __main__ import original_pscan_fn',
        globals={'A': A, 'X': X, 'Y_init': Y_init})
    
    
    #print(tref.timeit(100))
    print(t1.timeit(100))
    print(t0.timeit(100))
    print(t2.timeit(100))



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
    profile = False
    if profile:
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

    gA, gX, gY_init = torch.autograd.grad(s_, (A, X, Y_init), retain_graph=True)

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
