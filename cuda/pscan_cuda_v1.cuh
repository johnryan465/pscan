#include <torch/extension.h>


template <bool REVERSE>
torch::Tensor pscan_cuda_wrapper(torch::Tensor A, torch::Tensor X);