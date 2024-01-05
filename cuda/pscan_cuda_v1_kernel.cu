#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include "pscan_cuda_v1.cuh"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
struct PairScalar;
template <>
struct PairScalar<float>
{
    typedef float2 type;
};
template <>
struct PairScalar<double>
{
    typedef double2 type;
};

template <typename vec_t>
struct MultAddFunctor
{
    __device__ __forceinline__
        vec_t
        operator()(const vec_t &a, const vec_t &b) const
    {
        return {a.x * b.x, a.y * b.x + b.y};
    }
};

template <typename scalar_t, int n, int m>
__device__ __forceinline__ void transpose(scalar_t *A)
{
    if (n == 2 && m == 2)
    {
        scalar_t tmp = A[1];
        A[1] = A[2];
        A[2] = tmp;
    }
    else if (n == 4 && m == 2)
    {
        scalar_t tmp = A[1];
        A[1] = A[4];
        A[4] = A[2];
        A[2] = tmp;
        tmp = A[3];
        A[3] = A[5];
        A[5] = A[6];
        A[6] = tmp;
    }
}

template <
    typename scalar_t,
    int ITEMS_PER_THREAD,
    int BLOCK_THREADS,
    bool REVERSE>
__global__ void pscan_cuda_forward_kernel(
    scalar_t *A,
    scalar_t *X,
    int dim_size,
    int state_size)
{
    // block ID
    const int bidx = blockIdx.x;
    const int didx = blockIdx.y;
    const int tid = threadIdx.x;
    int offset;
    if constexpr(REVERSE){
        offset = bidx * dim_size * state_size + didx * state_size + (BLOCK_THREADS - tid) * ITEMS_PER_THREAD - 1;
    }
    else{
        offset = bidx * dim_size * state_size + didx * state_size + tid * ITEMS_PER_THREAD;
    }

    typedef typename PairScalar<scalar_t>::type pair_type;
    typedef cub::BlockScan<pair_type, BLOCK_THREADS> BlockScanT;
    using TempStorageT = typename BlockScanT::TempStorage;

    extern __shared__ char smem[];

    auto &temp_storage = reinterpret_cast<TempStorageT &>(smem);

    pair_type thread_data[ITEMS_PER_THREAD];
    scalar_t* thread_data_scalar = reinterpret_cast<scalar_t*>(thread_data);

    if constexpr(REVERSE){
        if ((tid+1) * ITEMS_PER_THREAD <= state_size){
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                thread_data_scalar[i] = A[offset - i];
            }
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                thread_data_scalar[i+ITEMS_PER_THREAD] = X[offset - i];
            }
        }
    }
    else{
        if ((tid+1) * ITEMS_PER_THREAD <= state_size){
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                thread_data_scalar[i] = A[offset + i];
            }
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                thread_data_scalar[i+ITEMS_PER_THREAD] = X[offset + i];
            }
        }
    }


    // Inplace transpose of thread_data for small fixed size
    transpose<scalar_t, ITEMS_PER_THREAD, 2>(thread_data_scalar);
    BlockScanT(temp_storage).InclusiveScan(thread_data, thread_data, MultAddFunctor<pair_type>());
    transpose<scalar_t, ITEMS_PER_THREAD, 2>(thread_data_scalar);

    if constexpr(REVERSE){
        if ((tid+1) * ITEMS_PER_THREAD <= state_size){
            /*#pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                A[offset - i] = thread_data_scalar[i];
            }*/
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                X[offset - i] = thread_data_scalar[i+ITEMS_PER_THREAD];
            }
        }
    }
    else{
        if ((tid+1) * ITEMS_PER_THREAD <= state_size){
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                A[offset + i] = thread_data_scalar[i];
            }
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i){
                X[offset + i] = thread_data_scalar[i+ITEMS_PER_THREAD];
            }
        }
    }
}
template <typename T, int BLOCK_THREADS, int ARCH>
constexpr std::size_t arch_bytes_size = sizeof(typename cub::BlockScan<T,BLOCK_THREADS,cub::BLOCK_SCAN_RAKING ,1,1,ARCH>::TempStorage);
template <typename T, int BLOCK_THREADS, int... Archs>
constexpr auto archs_max_bytes = (std::max)(
    {
        arch_bytes_size<T, BLOCK_THREADS, Archs>...,
    });

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

template <typename scalar_t>
__global__ void transposeNoBankConflicts(scalar_t *odata, const scalar_t *idata, const int stride)
{
    __shared__ scalar_t tile[TILE_DIM][TILE_DIM + 1];

    int offset = blockIdx.z * stride;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    if (x < width)
    {
#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if ((y + j) < height)
            {
                tile[threadIdx.y + j][threadIdx.x] = idata[offset + ((y + j) * width) + x];
            }
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height)
    {
#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if ((y + j) < width)
            {
                odata[offset + ((y + j) * height) + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

template <bool REVERSE>
torch::Tensor pscan_cuda_wrapper(torch::Tensor A, torch::Tensor X)
{
    const auto batch_size = A.size(0);
    const auto state_size = A.size(2);
    const auto dim_size = A.size(1);

    size_t const num_streams{4};
    const int offset = (batch_size / num_streams) * state_size * dim_size;

    std::vector<cudaStream_t> streams(num_streams);
    torch::Tensor X_ = torch::empty({X.size(0), X.size(2), X.size(1)}, X.options());

    for (size_t i = 0; i < num_streams; ++i){
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_transpose_cuda", ([&]
                                                                  {
        for (size_t i = 0; i < num_streams; ++i){
            dim3 dimGrid(dim_size/TILE_DIM, state_size/TILE_DIM, batch_size / num_streams);
            dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
            transposeNoBankConflicts<<<dimGrid, dimBlock, 0, streams[i]>>>(
                X_.data<scalar_t>()+ offset*i,
                X.data<scalar_t>() + offset*i,
                X.stride(0)
            );
        } }));

    const int threads = 1024;
    const int elements_per_thread = 2;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&]
                                                                {
    for(size_t i = 0; i < num_streams; ++i){
        const auto blocks = dim3(batch_size / num_streams, dim_size, 1);

        typedef typename PairScalar<scalar_t>::type pair_type;
        auto block_scan_temp_bytes = archs_max_bytes<pair_type, threads, 700, 800, 860>;
        auto smem_size = (std::max)(1 * sizeof(pair_type), block_scan_temp_bytes);
    
        pscan_cuda_forward_kernel<scalar_t, elements_per_thread, threads, REVERSE><<<blocks, threads, smem_size, streams[i]>>>(
            A.data<scalar_t>() + offset*i,
            X_.data<scalar_t>() + offset*i,
            dim_size,
            state_size
        );
    } }));

    for (size_t i = 0; i < num_streams; ++i)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    return X_;
}

template torch::Tensor pscan_cuda_wrapper<true>(torch::Tensor A, torch::Tensor X);
template torch::Tensor pscan_cuda_wrapper<false>(torch::Tensor A, torch::Tensor X);
