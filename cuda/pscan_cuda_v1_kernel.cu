#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>


#include <vector>


template <typename T> struct PairScalar;
template <> struct PairScalar<float>
{
    typedef float2 type;
};
template <> struct PairScalar<double>
{
    typedef double2 type;
};

template <typename vec_t>
struct MultAddFunctor
{
    __device__ __forceinline__
    vec_t operator()(const vec_t &a, const vec_t &b) const {
        return {a.x * b.x, a.y * b.x + b.y};
    }
};


template <
    typename scalar_t,
    int ITEMS_PER_THREAD,
    int BLOCK_THREADS
>
__global__ void pscan_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> X,
    int state_size)
     {
    // block ID
    const int bidx = blockIdx.y;
    const int didx = blockIdx.x;
    const int tid = threadIdx.x; 

    typedef typename PairScalar<scalar_t>::type pair_type;

    typedef cub::BlockScan<pair_type, BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Obtain a segment of consecutive items that are blocked across threads
    pair_type thread_data[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (i + tid * ITEMS_PER_THREAD < state_size){
            thread_data[i] = {A[bidx][didx][i + tid * ITEMS_PER_THREAD], X[bidx][didx][i + tid * ITEMS_PER_THREAD]};
        }
    }
    //BlockLoad(temp_storage.load).Load(custom_it_in, thread_data, state_size);
    BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, MultAddFunctor<pair_type>());
    // BlockStore(temp_storage.store).Store(custom_it_out, thread_data, state_size);
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (i + threadIdx.x * ITEMS_PER_THREAD < state_size){
            A[bidx][didx][i + tid * ITEMS_PER_THREAD] = thread_data[i].x;
            X[bidx][didx][i + tid * ITEMS_PER_THREAD] = thread_data[i].y;
        }
    }
}

torch::Tensor pscan_cuda_forward(torch::Tensor A, torch::Tensor X) {

  const auto batch_size = A.size(0);
  const auto state_size = A.size(2);
  const auto dim_size = A.size(1);


  const int threads = 1024;
  const int elements_per_thread = 2;
  const auto blocks = dim3(dim_size, batch_size, 1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();


  AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&] {
    pscan_cuda_forward_kernel<scalar_t, elements_per_thread, threads><<<blocks, threads, 0, stream>>>(
        A.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        X.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        state_size
    );
  }));

  return A;
}
