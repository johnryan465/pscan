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


torch::Tensor pscan_cuda_forward(torch::Tensor A, torch::Tensor B) {
  AT_DISPATCH_FLOATING_TYPES(A.type(), "pscan_forward_cuda", ([&] {

    typedef typename PairScalar<scalar_t>::type pair_type;

    pair_type* A_ptr = reinterpret_cast<pair_type*>(A.data<scalar_t>());
    int* B_ptr = B.data<int>();

    MultAddFunctor<pair_type> bin_op;

    // Determine temporary device storage requirements for inclusive prefix scan
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScanByKey(d_temp_storage, temp_storage_bytes, B_ptr, A_ptr, A_ptr, bin_op, B.numel(), cub::Equality());
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveScanByKey(d_temp_storage, temp_storage_bytes, B_ptr, A_ptr, A_ptr, bin_op, B.numel(), cub::Equality());
  }));

  return A;
}
