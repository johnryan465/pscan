import torch

import pscan_cuda_v1

class FastPScan(torch.autograd.Function):
    @staticmethod
    def pscan_fn(A, X):
        A_ = A[:,None,:].repeat(1, X.size(2), 1)
        A_, X_ = pscan_cuda_v1.forward(A_, X)
        return A_, X_

    @staticmethod
    def pscan_fn_backward(A, X):
        A_ = A[:,None,:].repeat(1, X.size(2), 1)
        A_, X_ = pscan_cuda_v1.backward(A_, X)
        return A_, X_   


    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A.clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star, ctx.X_star = FastPScan.pscan_fn(A, X)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star
        A = ctx.A
        R = grad_output.contiguous()
        A_ = torch.roll(A, -1, dims=1)
        _, R = FastPScan.pscan_fn_backward(A_, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)

fn = FastPScan.apply