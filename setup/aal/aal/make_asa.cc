/*
 * Cuda operators for asa packing
 */

#include <torch/extension.h>
#include <torch/torch.h>


using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;


std::tuple<Tensor, Tensor, IntArrayRef> make_asa_forward_cuda(Tensor data, bool relu);
Tensor make_asa_backward_cuda(Tensor grad_output, Tensor mask, IntArrayRef data_size, bool relu);

std::tuple<Tensor, Tensor, IntArrayRef> make_asa_forward(Tensor data, bool relu) 
{                                               
  return make_asa_forward_cuda(data, relu);
}

Tensor make_asa_backward(Tensor grad_output, Tensor mask, IntArrayRef data_size, bool relu) 
{  
  return make_asa_backward_cuda(grad_output, mask, data_size, relu);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_asa_forward", &make_asa_forward);
  m.def("make_asa_backward", &make_asa_backward);
  }
