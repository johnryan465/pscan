# Using a Fenwick tree inspired approach to solve the parralel scan problem:

# This formulation will hopefully be simplier to understand.

# We have an associative binary operation on (A_1, X_1), (A_2, X_2) -> (A_1 * A_2, A_1 * X_2 + X_1)

# We can use this to construct the fenwick tree in O(log(T)) time (parallel).
# We can then query this to construct the output in O(log(T)) time (parallel).

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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

    #for i in range(X_.shape[0]):
    #    for k in range(X_.shape[1]):
    current = (A_[:,:,0], X_[:,:,0])
    for j in range(1,X_.shape[2]):
        current = assoc_op(current, (A_[:,:,j], X_[:,:,j]))
        X_[:,:,j] = current[1]
        A_[:,:,j] = current[0]
    return A_.transpose(1, 2) * Y_init + X_.transpose(1, 2)

    
            


#pscan = fast_pscan #PScan.apply

######################################################################


if __name__ == "__main__":
    N, T, D = 2, 4, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_().cuda()
    X = torch.randn(N, T, D, dtype=torch.float64).requires_grad_().cuda()
    Y_init = torch.randn(N, D, dtype=torch.float64).requires_grad_().cuda()

    print("Testing...")
    print(fast_pscan(A, X, Y_init))
    print(slow_fenwick_pscan(A, X, Y_init))


