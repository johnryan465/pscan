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
    scalar_t* A,
    scalar_t* X,
    int dim_size,
    int state_size)
     {
    // block ID
    const int bidx = blockIdx.x;
    const int didx = blockIdx.y;
    const int tid = threadIdx.x; 
    const int offset = bidx * dim_size * state_size + didx * state_size + tid * ITEMS_PER_THREAD;

    typedef typename PairScalar<scalar_t>::type pair_type;
    typedef cub::BlockScan<pair_type, BLOCK_THREADS> BlockScanT;
    using TempStorageT = typename BlockScanT::TempStorage;

    extern __shared__ char smem[];
    
    auto& temp_storage = reinterpret_cast<TempStorageT&>(smem);

    pair_type thread_data[ITEMS_PER_THREAD];

    if (1 + tid * ITEMS_PER_THREAD < state_size){
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD / 2; ++i) {
            pair_type tmp = reinterpret_cast<pair_type*>(A+offset+i*2)[0];
            thread_data[i*2].x = tmp.x;
            thread_data[i*2 + 1].x = tmp.y;
        }
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD / 2; ++i) {
            pair_type tmp = reinterpret_cast<pair_type*>(X+offset+i*2)[0];
            thread_data[i*2].y = tmp.x;
            thread_data[i*2 + 1].y = tmp.y;
        }
    }
    BlockScanT(temp_storage).InclusiveScan(thread_data, thread_data, MultAddFunctor<pair_type>());
    if (1 + tid * ITEMS_PER_THREAD < state_size){
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD / 2; ++i) {
            pair_type tmp = reinterpret_cast<pair_type*>(A+offset+i*2)[0];
            reinterpret_cast<pair_type*>(A+offset+i*2)[0] = {thread_data[i*2].x, thread_data[i*2 + 1].x};
        }
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD / 2; ++i) {
            pair_type tmp = reinterpret_cast<pair_type*>(X+offset+i*2)[0];
            reinterpret_cast<pair_type*>(X+offset+i*2)[0] = {thread_data[i*2].y, thread_data[i*2 + 1].y};
        }
    }
}
template <typename T, int BLOCK_THREADS, int ARCH>
constexpr std::size_t arch_bytes_size = sizeof(
    typename cub::BlockScan<
        T,
        BLOCK_THREADS,
        cub::BLOCK_SCAN_RAKING /* ALGORITHM */,
        1 /* BLOCK_DIM_Y */,
        1 /* BLOCK_DIM_Z */,
        ARCH>::TempStorage
);
template <typename T, int BLOCK_THREADS, int... Archs>
constexpr auto archs_max_bytes = (std::max)(
    {arch_bytes_size<T, BLOCK_THREADS, Archs>...,});

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

template <typename scalar_t>    
__global__ void transposeNoBankConflicts(scalar_t *odata, const scalar_t *idata, const int stride)
{
    __shared__ scalar_t tile[TILE_DIM][TILE_DIM+1];


    int offset = blockIdx.z * stride;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    if(x < width){
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
            if((y+j) < height){
                tile[threadIdx.y+j][threadIdx.x] = idata[offset + ((y+j)*width) + x];
            }
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if(x < height){
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
            if((y+j) < width){
                odata[offset + ((y+j)*height) + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}



torch::Tensor pscan_cuda_forward(torch::Tensor A, torch::Tensor X) {
  const auto batch_size = A.size(0);
  const auto state_size = A.size(2);
  const auto dim_size = A.size(1);

  // Fast transpose
  
  torch::Tensor X_ = torch::empty({X.size(0), X.size(2), X.size(1)}, X.options());

  dim3 dimGrid(dim_size/TILE_DIM, state_size/TILE_DIM, batch_size);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);


    AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_transpose_cuda", ([&] {
            transposeNoBankConflicts<<<dimGrid, dimBlock>>>(
            X_.data<scalar_t>(),
            X.data<scalar_t>(),
            X.stride(0)
        );
    }));




  const int threads = 1024;
  const int elements_per_thread = 2;
  const auto blocks = dim3(batch_size, dim_size, 1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();


  AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&] {
    typedef typename PairScalar<scalar_t>::type pair_type;
    auto block_scan_temp_bytes = archs_max_bytes<pair_type, threads, 700, 800, 860>;
    auto smem_size = (std::max)(1 * sizeof(pair_type), block_scan_temp_bytes);
  
    pscan_cuda_forward_kernel<scalar_t, elements_per_thread, threads><<<blocks, threads, smem_size, stream>>>(
        A.data<scalar_t>(),
        X_.data<scalar_t>(),
        dim_size,
        state_size
    );
  }));

  return X_;
}
