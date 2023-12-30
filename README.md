# FastPSCAN


This implements 5 different algorithms for the parallel scan problem in pytorch. The algorithms are:
- Naive: The naive algorithm that uses a for loop to compute the scan.
- Original: Code written by https://twitter.com/francoisfleuret which motivated this project.
- Heinsen: The algorithm described in https://arxiv.org/abs/2311.06281
- CUDA_V1: The (currently fastest) algorithm which uses a custom CUDA kernel + CUB to do the reduction on a block level. **This algorithm will only work for shorter values of T.**
- CUDA_V2: This does a device level reduction using CUB, it is slower than CUDA_V1 but it is more general and works for much larger input size. **This is the recommended algorithm to use.**

## Benchmark

Benchmarking was done on a 3090 GPU.

### Forward
original_pscan_fn(A, X, Y_init)
setup: from __main__ import original_pscan_fn
  911.40 us
  1 measurement, 1000 runs , 1 thread


heinsen_pscan(A, X, Y_init)
setup: from __main__ import heinsen_pscan
  783.34 us
  1 measurement, 1000 runs , 1 thread


pscan_cuda_fn(A, X, Y_init)
setup: from __main__ import pscan_cuda_fn
  113.88 us
  1 measurement, 1000 runs , 1 thread


pscan_cuda_v2_fn(A, X, Y_init)
setup: from __main__ import pscan_cuda_v2_fn
  149.98 us
  1 measurement, 1000 runs , 1 thread

### Forward + Backward

backward_wrapper(pscan_cuda_fn,A, X, Y_init)
setup: from __main__ import pscan_cuda_fn, backward_wrapper
  394.81 us
  1 measurement, 1000 runs , 1 thread

backward_wrapper(heinsen_pscan, A, X, Y_init)
setup: from __main__ import heinsen_pscan, backward_wrapper
  1.63 ms
  1 measurement, 1000 runs , 1 thread

backward_wrapper(original_pscan_fn, A, X, Y_init)
setup: from __main__ import original_pscan_fn, backward_wrapper
  2.07 ms
  1 measurement, 1000 runs , 1 thread

backward_wrapper(pscan_cuda_v2_fn, A, X, Y_init)
setup: from __main__ import pscan_cuda_v2_fn, backward_wrapper
  453.88 us
  1 measurement, 1000 runs , 1 thread