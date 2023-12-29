from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pscan',
    ext_modules=[
        CUDAExtension('pscan_cuda', [
            'pscan_cuda.cpp',
            'pscan_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })