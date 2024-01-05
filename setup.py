from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cc_flag = []

cc_flag.append("-gencode")
cc_flag.append("arch=compute_70,code=sm_70")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
# cc_flag.append("-gencode")


setup(
    name='fastpscan',
    packages=['fastpscan'],
    package_dir={'fastpscan':'src'},
    ext_modules=[
        CUDAExtension(
            name='pscan_cuda_v1', 
            sources=[
            'cuda/pscan_cuda_v1.cpp',
            'cuda/pscan_cuda_v1_kernel.cu'
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "--threads",
                    "4",
                ]
                + cc_flag,
            }),
        CUDAExtension(
            name='pscan_cuda_v2', 
            sources=[
            'cuda/pscan_cuda_v2.cpp',
            'cuda/pscan_cuda_v2_kernel.cu'
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "--threads",
                    "4",
                ]
                + cc_flag,
            }),
        CUDAExtension(
            name='pscan_cuda_proger', 
            sources=[
                'cuda/cuscan_kernel.cu',
                'cuda/cuscan.cpp',
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "--threads",
                    "4",
                ]
                + cc_flag,
            }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })