/******************************************************************************
 * Based on code from https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan
 * Copyright (c) 2023, Tri Dao.
 * Apache License: https://github.com/state-spaces/mamba/blob/main/LICENSE (see copy in LICENSE.mamba)
 * Edited by Vol Kyrylov.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <torch/extension.h>

template<int kNThreads_, int kNItems_>
struct Kernel_traits {
    using input_t = float;
    using weight_t = float;
    using scan_t = float2;
    using vec_t = uint4; // For loading 4 items at a time
    static_assert(sizeof(vec_t) == 16);

    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static_assert(kNItems % 4 == 0);
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static constexpr int kNLoads = kNItems / kNElts;
    static_assert(kNItems % kNElts == 0);
    static_assert(kNLoads == 1);
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;

    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_DIRECT>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                    sizeof(typename BlockLoadVecT::TempStorage),
                                                    sizeof(typename BlockStoreT::TempStorage),
                                                    sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename scalar_t> struct FirstOrderScanOp;

template<>
struct FirstOrderScanOp<float> {
    __device__ __forceinline__ float2 operator()(const float2 &ab0, const float2 &ab1) const {
        return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
    }
};

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename scalar_t> struct FirstOrderScanPrefixCallbackOp {
    using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;
    // Constructor
    __device__ FirstOrderScanPrefixCallbackOp(scan_t running_prefix_) : running_prefix(running_prefix_) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        running_prefix = FirstOrderScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;
    }
};

struct Params {
    int batch;
    int dim;
    int seqlen;
    int n_chunks;
    signed long tokens_batch_stride;
    signed long tokens_d_stride;
    signed long gates_batch_stride;
    signed long gates_d_stride;
    signed long out_batch_stride;
    signed long out_d_stride;
    void* tokens_ptr;
    void* gates_ptr;
    void* x_ptr;
    void* out_ptr;
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void simple_scan_kernel(Params params) {
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;
    using vec_t = typename Ktraits::vec_t;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNLoads = Ktraits::kNLoads;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    input_t *tokens = reinterpret_cast<input_t *>(params.tokens_ptr) + batch_id * params.tokens_batch_stride
        + dim_id * params.tokens_d_stride;
    input_t *gates = reinterpret_cast<input_t *>(params.gates_ptr) + batch_id * params.gates_batch_stride
        + dim_id * params.gates_d_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks;

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t tokens_vals[kNItems], gates_vals[kNItems];
        __syncthreads();

        cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_DIRECT>(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(tokens),
            reinterpret_cast<vec_t(&)[kNLoads]>(tokens_vals)
        );
        tokens += kChunkSize;
        cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_DIRECT>(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(gates),
            reinterpret_cast<vec_t(&)[kNLoads]>(gates_vals)
        );
        gates += kChunkSize;

        float out_vals[kNItems];

        __syncthreads();
        scan_t thread_data[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            thread_data[i] = make_float2(gates_vals[i], tokens_vals[i]);
        }

        // Initialize running total
        // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
        scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? *smem_running_prefix : make_float2(1.f, 0.f);

        FirstOrderScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
        cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>(smem_scan).InclusiveScan(
            thread_data, thread_data, FirstOrderScanOp<weight_t>(), prefix_op
        );

        // There's a syncthreads in the scan op, so we don't need to sync here.
        // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
        if (threadIdx.x == 0) {
            *smem_running_prefix = prefix_op.running_prefix;
            x[chunk] = prefix_op.running_prefix;
        }

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            out_vals[i] = thread_data[i].y;
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();

        input_t write_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { write_vals[i] = out_vals[i]; }
        cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_DIRECT>(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(out),
            reinterpret_cast<vec_t(&)[kNLoads]>(write_vals)
        );
    }
}

template <int kNThreads, int kNItems>
void simple_scan_cuda(Params &params, cudaStream_t stream) {
    assert(params.seqlen % (kNThreads * kNItems) == 0);
    using Ktraits = Kernel_traits<kNThreads, kNItems>;
    auto kernel = &simple_scan_kernel<Ktraits>;

    constexpr int kSmemSize = Ktraits::kSmemSize + sizeof(typename Ktraits::scan_t);
    // printf("smem_size = %d\n", kSmemSize);
    dim3 grid(params.batch, params.dim);

    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }

    kernel<<<grid, kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor>
simple_scan_forward(const at::Tensor &gates, const at::Tensor &tokens) {
    auto input_type = tokens.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float);

    TORCH_CHECK(gates.scalar_type() == input_type);
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.stride(-1) == 1 || tokens.size(-1) == 1);
    TORCH_CHECK(gates.stride(-1) == 1 || gates.size(-1) == 1);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
    CHECK_SHAPE(tokens, batch_size, dim, seqlen);
    CHECK_SHAPE(gates, batch_size, dim, seqlen);

    // const int n_chunks = (seqlen + 512 - 1) / 512;
    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // at::Tensor out = torch::empty_like(tokens);
    // Right now tokens have BHL layout and gates has HBL layout, and we want out to have HBL layout
    at::Tensor out = torch::zeros_like(gates) + 42;
    at::Tensor x;
    x = torch::zeros({batch_size, dim, n_chunks, 2}, tokens.options().dtype(input_type));

    Params params = {
        .batch = batch_size,
        .dim = dim,
        .seqlen = seqlen,
        .n_chunks = n_chunks,
        .tokens_batch_stride = tokens.stride(0),
        .tokens_d_stride = tokens.stride(1),
        .gates_batch_stride = gates.stride(0),
        .gates_d_stride = gates.stride(1),
        .out_batch_stride = out.stride(0),
        .out_d_stride = out.stride(1),
        .tokens_ptr = tokens.data_ptr(),
        .gates_ptr = gates.data_ptr(),
        .x_ptr = x.data_ptr(),
        .out_ptr = out.data_ptr(),
    };

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)tokens.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // XXX: kNItems is hardcoded to 4, as we would like to always use BLOCK_STORE_DIRECT for now
    if (params.seqlen <= 128) {
        simple_scan_cuda<32, 4>(params, stream);
    } else if (params.seqlen <= 256) {
        simple_scan_cuda<64, 4>(params, stream);
    } else if (params.seqlen <= 512) {
        simple_scan_cuda<128, 4>(params, stream);
    } else if (params.seqlen <= 1024) {
        simple_scan_cuda<256, 4>(params, stream);
    } else {
        simple_scan_cuda<512, 4>(params, stream);
    }
    std::vector<at::Tensor> result = {out, out}; // A hack to make the shapes correct for benchmarking
    return result;
}
