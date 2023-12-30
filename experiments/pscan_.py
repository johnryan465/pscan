#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, math
import torch.nn.functional as F

######################################################################
from torch.profiler import profile, record_function, ProfilerActivity

import triton
import triton.language as tl


@triton.jit
def _mul_op(a, b):
    return a * b

@triton.jit
def expand_iter_(A, X):
    """
    Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    A'[i, t] = A[i, 2*t] * A[i, 2*t - 1]
    X'[i, t] = A[i, 2*t] * X[i, 2*t - 1] + X[i, 2*t]
    """
    # First we need to compute Y_T, the final value, through a scan

    pid = tl.program_id(axis=0)  # This thread is responsible for setting the value of Y[pid]

    # This program will process inputs that are offset from the initial data.

    #compute associative scan with A mulitplicaton


    max_offset = A.shape[1] // 2

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


def logcumsumexp(X_real, X_complex, dim=-1):
    X_sign = (((X_complex % 2) * (-2)) + 1)
    c = torch.cummax(X_real, dim=dim)[0]
    # c = 10
    #print("X_real", X_real.max())
    X_real_ = X_real - c
    # print(c)

    exp__ = torch.exp(X_real_)* X_sign


    # exp_ = exp_.to(torch.complex128)
    # exp_ = torch.complex(exp__, torch.ones_like(exp__) * eps)
    eps = 1e-10
    cumsum_ = torch.cumsum(exp__, dim=dim).to(torch.complex128)
    #print("exp", exp_)
    #print("cumsum", cumsum_)

    res = torch.log(cumsum_)  + c
    #print("res", res.nansum())
    return res # cumsum_  * torch.exp(c)

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

def assoc_op(A1, A2):
    return A2[0] * A1[0], (A2[0] * A1[1]) + A2[1]

def slow_fenwick_pscan(A, X, Y_init):
    A = A[:, :, None].repeat(1, 1, X.shape[-1])
    Y_init = Y_init[:, None,:]
    X_ = X.transpose(1, 2)
    A_ = A.transpose(1, 2)

    current = (A_[:,:,0], X_[:,:,0])
    for j in range(1,X_.shape[2]):
        # print(j)
        current = assoc_op(current, (A_[:,:,j], X_[:,:,j]))
        X_[:,:,j] = current[1]
        A_[:,:,j] = current[0]
    return A_.transpose(1, 2) * Y_init + X_.transpose(1, 2)


#pscan = fast_pscan #PScan.apply
# pscan = PScan.apply
pscan = slow_fenwick_pscan

######################################################################


if __name__ != "__main__":
    x_complex = torch.tensor([0, 1], dtype=torch.float64)
    inp = torch.tensor([1, 1], dtype=torch.float64)
    print(inp)

    print(torch.logcumsumexp(torch.complex(inp, torch.pi*x_complex), dim=-1))
    print(logcumsumexp(inp, x_complex, dim=-1))

if __name__ != "__main__":
    # test the two logcumsumexp implementations
    X_real = torch.randn(2, 3, 1, dtype=torch.float64)
    X_complex = torch.ones(2, 3, 1, dtype=torch.float64)
    A_real = torch.randn(2, 1, dtype=torch.float64)
    A_complex = torch.ones(2, 1, dtype=torch.float64)
    a_star_real = F.pad(torch.cumsum(A_real, dim=-1), (1,0))[:,None,:]            # eq (2) in paper
    a_star_complex = F.pad(torch.cumsum(A_complex, dim=-1), (1,0))[:,None,:]                   # eq (2) in paper

    log_x0_plus_b_star_1 = torch.exp(torch.logcumsumexp(torch.complex(X_real - a_star_real, torch.pi * (X_complex - a_star_complex).to(dtype=torch.float64)), dim=-1)) # logcumsumexp(X_real - a_star_real, X_complex - a_star_complex, dim=-1)  # eq (7) in paper
    log_x0_plus_b_star_2 = logcumsumexp(X_real - a_star_real, X_complex- a_star_complex, dim=-1) # logcumsumexp(X_real - a_star_real, X_complex - a_star_complex, dim=-1)  # eq (7) in paper

    print((log_x0_plus_b_star_1 - log_x0_plus_b_star_2).norm())
    print(log_x0_plus_b_star_1)
    print(log_x0_plus_b_star_2)

    print(log_x0_plus_b_star_1.shape)
    print(log_x0_plus_b_star_2.shape)


if __name__ != "__main__":
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
        Y = pscan(A, X, Y_init)
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    duration = time.perf_counter() - start_time
    print(f"duration {duration}")

    s_ = Y.sum()

    gA, gX, gY_init = torch.autograd.grad(s_, (A, X, Y_init), retain_graph=True)

    print("gA", gA)
    print("gA_ref", gA_ref)
    print("gA", (gA - gA_ref).norm())

    diff = gA - gA_ref

    # find the top 10 largest values in diff
    print(diff.flatten().abs().topk(10))

    print(s)
    print(s_)


    # print(gX)
    # print(gY_init)

    print((gA - gA_ref).norm())
    print((gX - gX_ref).norm())
    print((gY_init - gY_init_ref).norm())

    Y1 = pscan(A[:, : T // 2], X[:, : T // 2], Y_init)
    Y2 = pscan(A[:, T // 2 :], X[:, T // 2 :], Y1[:, -1])

    print((Y - torch.cat([Y1, Y2], dim=1)).norm())
