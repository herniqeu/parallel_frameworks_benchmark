from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import os
import subprocess

# Define CUDA paths
CUDA_PATH = os.getenv('CUDA_PATH', '/usr/local/cuda-10.1')
CUDA_INCLUDE_PATH = os.path.join(CUDA_PATH, 'include')
CUDA_LIB_PATH = os.path.join(CUDA_PATH, 'lib64')

class CUDA_build_ext(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if hasattr(ext, 'cuda_sources'):
                cuda_objects = self.compile_cuda_sources(ext)
                ext.extra_objects.extend(cuda_objects)
        build_ext.build_extensions(self)

    def compile_cuda_sources(self, ext):
        cuda_objects = []
        build_temp = os.path.join(self.build_temp, 'cuda_objects')
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        for cuda_source in ext.cuda_sources:
            object_name = os.path.splitext(os.path.basename(cuda_source))[0] + '.o'
            object_file = os.path.join(build_temp, object_name)

            nvcc_cmd = [
                'nvcc',
                '-c',
                '-o', object_file,
                '-Xcompiler', '-fPIC',
                '--shared',
                '-arch=sm_35',
                cuda_source
            ]
            
            if not self.dry_run:
                subprocess.check_call(nvcc_cmd)
            
            cuda_objects.append(object_file)
        return cuda_objects

# Define extensions
extensions = [
    Extension(
        "implementations.cuda.matrix_multiplication",
        sources=["implementations/cuda/matrix_multiplication_wrapper.cpp"],
        include_dirs=[np.get_include(), CUDA_INCLUDE_PATH],
        library_dirs=[CUDA_LIB_PATH],
        libraries=['cudart', 'cuda'],
        extra_compile_args=['-O3'],
        extra_link_args=[f'-Wl,-rpath,{CUDA_LIB_PATH}'],
        cuda_sources=["implementations/cuda/matrix_multiplication.cu"],
    ),
    Extension(
        "implementations.openmp.matrix_multiplication",
        ["implementations/openmp/matrix_multiplication.cpp"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "implementations.openmp.monte_carlo",
        ["implementations/openmp/monte_carlo.cpp"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "implementations.openmp.mandelbrot",
        ["implementations/openmp/mandelbrot.cpp"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name="benchmark_implementations",
    version="0.1.0",
    packages=['implementations', 'implementations.openmp', 
              'implementations.cuda', 'implementations.cupy'],
    ext_modules=extensions,
    cmdclass={'build_ext': CUDA_build_ext},
    install_requires=[
        'numpy>=1.21.0',
        'cupy-cuda101>=7.8.0',
        'psutil>=5.8.0',
        'gputil>=1.4.0',
    ],
)