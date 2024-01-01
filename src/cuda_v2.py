import torch

import pscan_cuda_v2

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
        A = A.repeat(1, 1, X.size(2))
        shape = (X.shape[0], X.shape[2], X.shape[1])
        C = torch.stack([A, X], dim=3).transpose(1,2).contiguous()
        C = C.view(-1, shape[-1], 2)
        B = torch.arange(C.size(0), device=A.device, dtype=torch.int).view(-1, 1).repeat(1, C.size(1))
        C = pscan_cuda_v2.forward(C, B)
        C_ = C.view(shape + (2,)).transpose(1, 2)
        A_ = C_[:,:,:,0]
        X_ = C_[:,:,:,1]
        return A_, X_

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

    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star
        A = ctx.A
        R = grad_output
        A_ = torch.cat([A[:, 1:],torch.ones_like(A[:, :1])], dim=1)
        A__ = torch.flip(A_, [1])
        rev_r = torch.flip(R, [1])
        _, R_ = FastPScan.pscan_fn(A__, rev_r)
        R = torch.flip(R_, [1])
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)

fn = FastPScan.apply