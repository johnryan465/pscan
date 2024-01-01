import torch
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity

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


if __name__ != "__main__":
    N, T, D = 2, 1047, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_().cuda()
    X = torch.rand(N, T, D, dtype=torch.float64).requires_grad_().cuda()
    Y_init = torch.rand(N, D, dtype=torch.float64).requires_grad_().cuda()

    gA_ref, gX_ref, gY_init_ref = backward_wrapper(naive_pscan, A, X, Y_init)
    gA_og, gX_og, gY_init_og = backward_wrapper(original_pscan_fn, A, X, Y_init)
    gA_cuda, gX_cuda, gY_init_cuda = backward_wrapper(pscan_cuda_fn, A, X, Y_init)
    gA_heinsen, gX_heinsen, gY_init_heinsen = backward_wrapper(heinsen_pscan, A, X, Y_init)
    gA_cuda_v2, gX_cuda_v2, gY_init_cuda_v2 = backward_wrapper(pscan_cuda_v2_fn, A, X, Y_init)
    gA_cuda_v3, gX_cuda_v3, gY_init_cuda_v3 = backward_wrapper(pscan_cuda_v3_fn, A, X, Y_init)
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #    for i in range(100):
    #        Y = pscan_cuda_v2_fn(A, X, Y_init)

    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    #if False:
    print("Testing Naive vs Original")
    assert torch.allclose(gA_ref, gA_og)
    assert torch.allclose(gX_ref, gX_og)
    assert torch.allclose(gY_init_ref, gY_init_og)
    print("Success!")

    print("Testing Naive vs CUDA")  
    assert torch.allclose(gA_ref, gA_cuda)
    assert torch.allclose(gX_ref, gX_cuda)
    assert torch.allclose(gY_init_ref, gY_init_cuda)
    print("Success!")


    print("Testing Naive vs Heinsen")
    assert torch.allclose(gA_ref, gA_heinsen)
    assert torch.allclose(gX_ref, gX_heinsen)
    assert torch.allclose(gY_init_ref, gY_init_heinsen)
    print("Success!")

    print("Testing Naive vs CUDA v2")
    assert torch.allclose(gA_ref, gA_cuda_v2)
    assert torch.allclose(gX_ref, gX_cuda_v2)
    assert torch.allclose(gY_init_ref, gY_init_cuda_v2)
    print("Success!")

    print("Testing Naive vs CUDA v3")
    assert torch.allclose(gA_ref, gA_cuda_v3)
    assert torch.allclose(gX_ref, gX_cuda_v3)
    assert torch.allclose(gY_init_ref, gY_init_cuda_v3)
    print("Success!")



if __name__ == "__main__":
    N, T, D = 128, 256, 256

    A = torch.rand(N, T, dtype=torch.float32).requires_grad_().cuda()
    X = torch.rand(N, T, D, dtype=torch.float32).requires_grad_().cuda()
    # print(A, X)
    Y_init = torch.rand(N, D, dtype=torch.float32).requires_grad_().cuda()

    # res_ref = naive_pscan(A, X, Y_init)
    res_og = original_pscan_fn(A, X, Y_init)
    res_cuda = pscan_cuda_fn(A, X, Y_init)
    # res_heinsen = heinsen_pscan(A, X, Y_init)
    #res_cuda_v2 = pscan_cuda_v2_fn(A, X, Y_init)
    # res_cuda_v3 = pscan_cuda_v3_fn(A, X, Y_init)

    print("Testing Original vs CUDA")
    # print(res_og)  
    # print(res_cuda)
    assert torch.allclose(res_og, res_cuda)
    #print("Success!")


    #print("Testing Original vs CUDA V2")
    #print(res_og)
    #print(res_cuda_v2)
    #assert torch.allclose(res_og, res_cuda_v2)
    #print("Success!")

    # print("Testing Original vs CUDA V3")
    #print(res_og)
    #print(res_cuda_v3)
    #assert torch.allclose(res_og, res_cuda_v3)
    #print("Success!")

