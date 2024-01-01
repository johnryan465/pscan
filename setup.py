from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastpscan',
    packages=['fastpscan'],
    package_dir={'fastpscan':'src'},
    ext_modules=[
        CUDAExtension('pscan_cuda_v1', [
            'cuda/pscan_cuda_v1.cpp',
            'cuda/pscan_cuda_v1_kernel.cu',
        ]),
        CUDAExtension(
            name='pscan_cuda_v2', 
            sources=[
            'cuda/pscan_cuda_v2.cpp',
            'cuda/pscan_cuda_v2_kernel.cu'
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })