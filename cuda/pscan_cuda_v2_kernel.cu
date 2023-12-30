#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>


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
    scalar_t* __restrict__ A_,
    int state_size)
     {
    // block ID
    const int bidx = blockIdx.x;
    //printf("bidx: %d\n", bidx);

    typedef typename PairScalar<scalar_t>::type pair_type;

    pair_type* A = reinterpret_cast<pair_type*>(A_);
    
    typedef cub::BlockLoad<pair_type, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>   BlockLoad;
    typedef cub::BlockStore<pair_type, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_DIRECT>  BlockStore;
    typedef cub::BlockScan<pair_type, BLOCK_THREADS> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ union {
        typename BlockLoad::TempStorage     load;
        typename BlockScan::TempStorage     scan;
        typename BlockStore::TempStorage    store;
    } temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    pair_type thread_data[ITEMS_PER_THREAD];
    int block_offset = bidx * state_size; // * sizeof(pair_type);
    //printf("block_offset, bidx, state_size: %d, %d, %d\n", block_offset, bidx, state_size);
    int valid_items = state_size;

    BlockLoad(temp_storage.load).Load(A + block_offset, thread_data, valid_items);
    BlockScan(temp_storage.scan).InclusiveScan(thread_data, thread_data, MultAddFunctor<pair_type>());
    BlockStore(temp_storage.store).Store(A + block_offset, thread_data, valid_items);
}

torch::Tensor pscan_cuda_forward(torch::Tensor A) {

  const auto batch_size = A.size(0);
  const auto state_size = A.size(1);


  const int threads = 512;
  const int blocks = batch_size;

  AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&] {
    pscan_cuda_forward_kernel<scalar_t, 4, threads><<<blocks, threads>>>(
        A.data<scalar_t>(),
        state_size
    );
  }));

  return A;
}
