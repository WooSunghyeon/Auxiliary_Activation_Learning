/*
 * Packbits and upackbits for 1bits for ASA.
 */

#include <torch/extension.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <c10/cuda/CUDAGuard.h>

#define BLOCK_Y_DIM_MAX ((((int64_t)(1)) << 16) - 1)
#define fmax(a, b) ((a) > (b) ? (a): (b))

#define NUM_THREADS 512

using torch::IntArrayRef;
using torch::Tensor;


// Compute sign tensor and 1-bit activations (mask) and pack the mask into int32 streams
template <typename scalar_t>
__global__ void make_asa_forward_kernel(const scalar_t* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  scalar_t* __restrict__ output,
                                                  int64_t N,
                                                  int64_t mask_len,
                                                  bool relu) {
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t global_offset = (int64_t)blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = NUM_THREADS / (sizeof(int32_t) * 8);
  __shared__ int mask_shared[NUM_THREADS / (sizeof(int32_t) * 8)];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  if (id < N) {
    bool bit = data[id] > 0;
    if (bit) {
      output[id] = 1.0;
    } else {
      if (relu){
      output[id] = 0.0;
      }
      else{
      output[id] = -1.0;
      }
    }

    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

std::tuple<Tensor, Tensor, IntArrayRef> make_asa_forward_cuda(Tensor data, bool relu) {
   at::cuda::CUDAGuard device_guard(data.device());
  
  int64_t n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int64_t mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);
  Tensor output = torch::empty_like(data);
  IntArrayRef data_size = data.sizes();

  int threads = NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "make_asa_forward", ([&] {
    make_asa_forward_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
      n_elements, mask_len, relu);
  }));

  return std::make_tuple(output, mask, data_size);
}

// Unpack 1-bit sign activations from the saved int32 stream
template <typename scalar_t>
__global__ void make_asa_backward_kernel(int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ output,
                                                   int N, bool relu) {
  int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t global_offset = (int64_t)blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = NUM_THREADS / (sizeof(int32_t) * 8);

  if (id < N) {
    bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    if (bit) {
      output[id] = 1.0;
    } else {
      if (relu){
      output[id] = 0.0;
      }
      else{
      output[id] = -1.0;
      }
    }
  }
}


Tensor make_asa_backward_cuda(Tensor grad_output, Tensor mask, IntArrayRef data_size, bool relu) {
  
   at::cuda::CUDAGuard device_guard(grad_output.device());
  
  auto options = torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device());
  Tensor unpack = torch::empty(data_size, options);

  int64_t n_elements = 1;
  for (size_t i = 0; i < unpack.dim(); ++i) {
    n_elements *= data_size[i];
  }

  int threads = NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "make_asa_backward", ([&] {
      make_asa_backward_kernel<scalar_t><<<blocks, threads>>>(
        mask.data_ptr<int32_t>(), unpack.data_ptr<scalar_t>(),
        n_elements, relu);
  }));

  return unpack;
}


