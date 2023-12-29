from torch.utils.cpp_extension import load
pscan = load( 'pscan', ['pscan_cuda.cpp', 'pscan_cuda_kernel.cu'], verbose=True)
help(pscan)