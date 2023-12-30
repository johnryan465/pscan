from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastpscan',
    packages=['fastpscan'],
    package_dir={'fastpscan':'src'},
    ext_modules=[
        CUDAExtension('pscan_cuda', [
            'cuda/pscan_cuda.cpp',
            'cuda/pscan_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })