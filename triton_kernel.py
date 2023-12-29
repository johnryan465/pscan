import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _add_op(a, b):
    return a + b


@triton.jit
def kernel_prefixsum(
    values,
    Z,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr
):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values_tensor = tl.load(values + offsets, mask=mask)
    logs = tl.log(values_tensor)
    # vf = bit_merge(values_tensor, factors_tensor)
    out_values, _ = tl.associative_scan(logs, 0, combine_fn=_add_op)
    
    tl.store(Z + offsets, out_values, mask=mask)


def prefixsum(
    A: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        A: [B, T] float32 values
        X: [B, T, D] float32 values
        Y_init: [B, D] float32 values

    Returns:
        Parallel scan of with A as coefficients and X as values
    """
    shape = A.shape
    result = torch.empty_like(A)

    grid = lambda meta: (triton.cdiv(shape[0], meta['BLOCK_SIZE']), )


    kernel_prefixsum[grid](
        A,
        result,
        shape[0],
        BLOCK_SIZE=1024
    )
    return result
    
if __name__ == "__main__":
    print(prefixsum(torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32).cuda()))