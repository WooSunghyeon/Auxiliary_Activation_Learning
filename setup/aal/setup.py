from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='aal',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'aal.cudnn_conv',
              ['aal/cudnn_conv.cpp']
          ),
          cpp_extension.CUDAExtension(
              'aal.batchnorm',
              ['aal/batchnorm.cpp']
          ),
          cpp_extension.CUDAExtension(
              'aal.make_asa',
              ['aal/make_asa.cc', 'aal/make_asa_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'aal.make_relu',
              ['aal/make_relu.cc', 'aal/make_relu_cuda_kernel.cu']
          )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
