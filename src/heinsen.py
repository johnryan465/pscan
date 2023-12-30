import torch
import torch.nn.functional as F
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


fn = fast_pscan