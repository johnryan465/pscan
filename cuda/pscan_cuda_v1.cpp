#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor pscan_cuda_forward(torch::Tensor A, torch::Tensor X);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> pscan_forward(torch::Tensor A, torch::Tensor X) {
  CHECK_INPUT(A);
  CHECK_INPUT(X);
  X = pscan_cuda_forward(A, X);
  A = A.transpose(1,2);
  X = X.transpose(1,2);
  return {A, X};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pscan_forward, "PScan forward (CUDA)");
}