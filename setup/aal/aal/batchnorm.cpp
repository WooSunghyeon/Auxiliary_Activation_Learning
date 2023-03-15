#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

/*
PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:
    - cudnn_batch_norm
    - cudnn_batch_norm_backward
*/

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon){
    
    return at::cudnn_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    double epsilon,
    const at::Tensor& reserveSpace){

    return at::cudnn_batch_norm_backward(
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        epsilon,
        reserveSpace);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_norm", &batch_norm, "batch norm");
    m.def("batch_norm_backward", &batch_norm_backward, "batch norm backward");
}