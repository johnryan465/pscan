#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>


#include <vector>

struct MultAddFunctor
{
    __device__ __forceinline__
    float2 operator()(const float2 &a, const float2 &b) const {
        return {a.x * b.x, a.y * b.x + b.y};
    }
};

template <typename scalar_t>
__global__ void pscan_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> A,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> X) {
    // thread ID
    const int tidx = threadIdx.x;
    //const int tidy = threadIdx.y;
    // block ID
    const int bidx = blockIdx.x;

    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<float2, 128> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    float2 thread_data[4] = {
        {A[bidx][tidx*4], X[bidx][tidx*4]},
        {A[bidx][tidx*4+1], X[bidx][tidx*4+1]},
        {A[bidx][tidx*4+2], X[bidx][tidx*4+2]},
        {A[bidx][tidx*4+3], X[bidx][tidx*4+3]}
    };
    //...
    // Collectively compute the block-wide inclusive prefix max scan
    BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, MultAddFunctor());
    //...
    // 
}

std::vector<torch::Tensor> pscan_cuda_forward(
    torch::Tensor A,
    torch::Tensor X) {

  const auto batch_size = X.size(0);
  const auto state_size = X.size(1);


  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&] {
    pscan_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        X.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
    );
  }));

  return {A, X};
}
