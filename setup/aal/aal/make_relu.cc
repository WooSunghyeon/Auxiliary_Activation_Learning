/*
 * Cuda operators for relu packing
 */

#include <torch/extension.h>
#include <torch/torch.h>


using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;


std::tuple<Tensor, Tensor, IntArrayRef> make_relu_forward_cuda(Tensor data);
Tensor make_relu_backward_cuda(Tensor grad_output, Tensor mask, IntArrayRef data_size);

std::tuple<Tensor, Tensor, IntArrayRef> make_relu_forward(Tensor data) 
{                                               
  return make_relu_forward_cuda(data);
}

Tensor make_relu_backward(Tensor grad_output, Tensor mask, IntArrayRef data_size) 
{  
  return make_relu_backward_cuda(grad_output, mask, data_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_relu_forward", &make_relu_forward);
  m.def("make_relu_backward", &make_relu_backward);
  }
