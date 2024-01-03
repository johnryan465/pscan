/******************************************************************************
 * Based on code from https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan
 * Copyright (c) 2023, Tri Dao.
 * Apache License: https://github.com/state-spaces/mamba/blob/main/LICENSE (see copy in LICENSE.mamba)
 * Edited by Vol Kyrylov.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>


#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <torch/extension.h>


std::vector<at::Tensor>simple_scan_forward(const at::Tensor &gates, const at::Tensor &tokens) ;

std::vector<at::Tensor> simple_scan_cuda(const at::Tensor &gates, const at::Tensor &tokens) {
    return simple_scan_forward(gates, tokens);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &simple_scan_forward, "PScan forward (CUDA)");
}