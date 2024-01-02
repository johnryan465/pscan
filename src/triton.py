"""
This ccode is adapted from https://github.com/andreaskoepf/pscan_kernel/blob/main/pscan_triton_fused_full.py
Full credit goes to Andreas Koepf
"""

import torch
import triton
import triton.language as tl


def cdiv(x, div):
    return (x + div - 1) // div


@triton.jit
def expand_kernel(
    A,  # [N, T, 1]
    X,  # [N, T, D]
    stride_AN: int,
    stride_AT: int,
    stride_AD: int,
    stride_XN: int,
    stride_XT: int,
    stride_XD: int,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    T_block_size: tl.constexpr,
    D_block_size: tl.constexpr,
):
    n = tl.program_id(axis=0)
    dim_chunk = tl.program_id(axis=1)

    A_base = A + n * stride_AN + stride_AD * dim_chunk
    X_base = X + n * stride_XN

    offs_dim = tl.arange(0, D_block_size) + dim_chunk * D_block_size

    view_stride = 1
    view_offset = 0
    while view_offset + view_stride < seqlen:
        indices0 = tl.arange(0, T_block_size) * 2 * view_stride
        indices1 = indices0 + view_stride

        tl.debug_barrier()

        block_offset = view_offset
        while block_offset < seqlen:
            # read values
            a1 = tl.load(
                A_base + (indices1 + block_offset) * stride_AT,
                mask=(indices1 + block_offset) < seqlen,
            )

            # load block T_block_size x D_block_size
            # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            x0 = tl.load(
                X_base
                + (indices0 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                mask=((indices0 + block_offset)[:, None] < seqlen)
                & (offs_dim[None, :] < dim),
            )
            x1 = tl.load(
                X_base
                + (indices1 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                mask=((indices1 + block_offset)[:, None] < seqlen)
                & (offs_dim[None, :] < dim),
            )
            x1 += x0 * a1[:, None]
            tl.store(
                X_base
                + (indices1 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                x1,
                mask=((indices1 + block_offset)[:, None] < seqlen)
                & (offs_dim[None, :] < dim),
            )

            tl.debug_barrier()

            # Aa[:, :, 1].mul_(Aa[:, :, 0])
            a0 = tl.load(
                A_base + (indices0 + block_offset) * stride_AT,
                mask=(indices0 + block_offset) < seqlen,
            )
            b = a0 * a1

            # store
            tl.store(
                A_base + (indices1 + block_offset) * stride_AT,
                b,
                mask=(indices1 + block_offset) < seqlen,
            )

            block_offset += T_block_size * view_stride * 2

        view_offset += view_stride
        view_stride = view_stride * 2

    view_stride = view_stride // 2
    view_offset -= view_stride

    # downward pass
    while view_stride > 0:
        indices1 = tl.arange(0, T_block_size) * 2 * view_stride + view_stride
        indices0 = indices1 + view_stride

        #tl.debug_barrier()

        block_offset = view_offset
        while block_offset < seqlen:
            # read values
            a0 = tl.load(
                A_base + (indices0 + block_offset) * stride_AT,
                mask=(indices0 + block_offset) < seqlen,
            )

            # Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
            x0 = tl.load(
                X_base
                + (indices0 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                mask=((indices0 + block_offset)[:, None] < seqlen)
                & (offs_dim[None, :] < dim),
            )
            x1 = tl.load(
                X_base
                + (indices1 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                mask=((indices1 + block_offset)[:, None] < seqlen)
                & (offs_dim[None, :] < dim),
            )
            x0 += x1 * a0[:, None]

            tl.store(
                X_base
                + (indices0 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                x0,
                mask=((indices0 + block_offset)[:, None] < seqlen)
                & (offs_dim[None, :] < dim),
            )

            tl.debug_barrier()

            # Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
            a1 = tl.load(
                A_base + (indices1 + block_offset) * stride_AT,
                mask=(indices1 + block_offset) < seqlen,
            )
            b = a0 * a1

            # store
            tl.store(
                A_base + (indices0 + block_offset) * stride_AT,
                b,
                mask=(indices0 + block_offset) < seqlen,
            )

            block_offset += T_block_size * view_stride * 2

        view_stride = view_stride // 2
        view_offset -= view_stride


def nextPowerOfTwo(x: int) -> int:
	power = 1
	while (power < x):
		power *= 2
	return power


def expand_triton(
    A: torch.Tensor,  # [N, T, 1]
    X: torch.Tensor,  # [N, T, D]
):
    # shape checks
    N, T, D = X.shape

    assert A.shape[0] == N, "N mismatch"
    assert A.shape[1] == T, "T mismatch"
    assert T == nextPowerOfTwo(T), "only pow2 vaues for T tested"

    if D >= 128:
        block_size_dim = 128
    elif D >= 64:
        block_size_dim = 64
    elif D >= 32:
        block_size_dim = 32
    elif D >= 16:
        block_size_dim = 16
    else:
        block_size_dim = 8

    if T >= 64:
        block_size_seq = 64
    elif T >= 32:
        block_size_seq = 32
    elif T >= 16:
        block_size_seq = 16
    else:
        block_size_seq = 8

    dim_blocks = cdiv(D, block_size_dim)

    # temporary expansion of A for temp storage
    A_ = A.repeat(1, 1, dim_blocks).contiguous()

    grid = (N, dim_blocks)
    expand_kernel[grid](
        A_,  # [N, T, dim_blocks]
        X,  # [N, T, D]
        stride_AN=A_.stride(0),
        stride_AT=A_.stride(1),
        stride_AD=A_.stride(2),
        stride_XN=X.stride(0),
        stride_XT=X.stride(1),
        stride_XD=X.stride(2),
        seqlen=T,
        dim=D,
        T_block_size=block_size_seq,
        D_block_size=block_size_dim,
    )
    A.copy_(A_[:, :, :1])




import torch

import pscan_cuda_v1

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
        A = A.repeat(1, 1, X.size(2)).transpose(1,2).contiguous()
        X = X.transpose(1,2).contiguous()
        #C = torch.stack([A, X], dim=3).transpose(1,2).contiguous()
        pscan_cuda_v1.forward(A, X)
        A_ = A.transpose(1,2)
        X_ = X.transpose(1,2)
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
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        # ctx.A_star, ctx.X_star = expand_triton(ctx.A, X)
        expand_triton(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star
        A = ctx.A
        R = grad_output
        A_ = torch.flip(A, [1])
        A__ = torch.cat([torch.ones_like(A_[:, -1:]), A_[:, :-1]], dim=1)
        rev_r = torch.flip(R, [1])
        _, R_ = FastPScan.pscan_fn(A__, rev_r)
        R = torch.flip(R_, [1])
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)

fn = FastPScan.apply