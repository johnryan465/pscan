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

template<typename T>
class CustomIterator {
public:
    // Constructor with a pointer to the start of the data and stride
    __host__ __device__  CustomIterator(T* ptr, int stride = 1) : ptr_(ptr), stride_(stride) {}

    __device__ T& operator*() {
        return *ptr_;
    }

    __device__ CustomIterator operator+(int offset) {
        // Create a new iterator that is offset from the current one
        return CustomIterator(ptr_ + offset * stride_, stride_);
    }

    __device__ CustomIterator& operator++() {
        ptr_ += stride_;
        return *this;
    }

    __device__ CustomIterator operator++(int) {
        CustomIterator old = *this;
        ptr_ += stride_;
        return old;
    }

    __device__ T& operator[](int index) {
        return *(ptr_ + index * stride_);
    }

private:
    T* ptr_;    // Pointer to the current element
    int stride_; // Custom stride
};

template <
    typename scalar_t,
    int ITEMS_PER_THREAD,
    int BLOCK_THREADS
>
__global__ void pscan_cuda_forward_kernel(
    scalar_t* __restrict__ A_,
    int state_size,
    int dim_size)
     {
    // block ID
    const int bidx = blockIdx.x;
    const int didx = blockIdx.y;

    typedef typename PairScalar<scalar_t>::type pair_type;

    pair_type* A = reinterpret_cast<pair_type*>(A_);
    int block_offset = (bidx * state_size * dim_size) + didx;
    CustomIterator<pair_type> custom_it_in(A + block_offset, dim_size);
    CustomIterator<pair_type> custom_it_out(A + block_offset, dim_size);

    
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
    BlockLoad(temp_storage.load).Load(custom_it_in, thread_data, state_size);
    BlockScan(temp_storage.scan).InclusiveScan(thread_data, thread_data, MultAddFunctor<pair_type>());
    BlockStore(temp_storage.store).Store(custom_it_out, thread_data, state_size);
}

torch::Tensor pscan_cuda_forward(torch::Tensor A) {

  const auto batch_size = A.size(0);
  const auto state_size = A.size(1);
  const auto dim_size = A.size(2);


  const int threads = 512;
  const auto blocks = dim3(batch_size, dim_size, 1);

  AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&] {
    pscan_cuda_forward_kernel<scalar_t, 4, threads><<<blocks, threads>>>(
        A.data<scalar_t>(),
        state_size,
        dim_size
    );
  }));

  return A;
}
