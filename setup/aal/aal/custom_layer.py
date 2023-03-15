import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import aal.cudnn_conv as cudnn_convolution
import aal.batchnorm as cudnn_batch_norm
import aal.make_relu as make_relu

'''
We make custom batchnorm layer.
Batchnorm2d can replace nn.BatchNorm2d
'''
class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.momentum = momentum
        self.one_minus_momentum = 1-momentum
    
    def forward(self, input):
        self._check_input_dim(input)
        return batch_norm2d.apply(input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.track_running_stats, self.momentum, self.eps)

class batch_norm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, track_running_stats, momentum, eps):
        output, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)
        ctx.save_for_backward(input, weight, running_mean, running_var, save_mean, save_var, reservedspace)
        ctx.eps =eps
        return output 
    
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        input, weight, running_mean, running_var, save_mean, save_var, reservedspace = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = cudnn_batch_norm.batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None

def batchnorm_forward(input, weight, bias, training, track_running_stats, running_mean, running_var, momentum, eps, backward = False):
    N = input.size(1) # channel
    input = input.permute(0,2,3,1).contiguous()
    input_shape = input.shape
    input = input.view(-1,N)
    if training:
        mu = input.mean(0)
        var = torch.var(input,0, unbiased=False)
        if track_running_stats and not(backward):
            running_mean.data = running_mean.mul(1-momentum).add(mu.mul(momentum)).data
            running_var.data = running_var.mul(1-momentum).add(var.mul(momentum)).data
        sqrt = torch.sqrt(var+eps).reciprocal()
        mu = mu.mul(sqrt)
        weight_div_sqrt = weight.mul(sqrt)
        y = input * weight_div_sqrt + bias.add(-mu*weight)
        return y.view(input_shape).permute(0,3,1,2).contiguous(), mu, weight_div_sqrt, sqrt
        
    else:
        y = input * weight.div(torch.sqrt(running_var+eps)) \
            + bias.add(-running_mean.div(torch.sqrt(running_var+eps)).mul(weight))
        return y.view(input_shape).permute(0,3,1,2).contiguous(), None, None, None


def batchnorm_backward(out, weight, bias, grad_output, mu_div_sqrt, weight_div_sqrt, sqrt, approximate_input = False):
    N = out.size(1) # channel
    out = out.permute(0,2,3,1).contiguous()
    out = out.view(-1,N)
    
    if approximate_input:
        out *= sqrt
        out -= mu_div_sqrt
    else:
        out -= bias
        out /= weight
        
    grad_out = grad_output.permute(0,2,3,1).contiguous()
    grad_shape = grad_out.shape
    grad_out = grad_out.view(-1, N)

    grad_bias = torch.sum(grad_out, 0)
    grad_weight = torch.sum(out*grad_out, 0)
    grad_input = weight_div_sqrt*(grad_out - grad_weight*out/grad_out.size(0) - grad_bias/grad_out.size(0) )
    
    grad_input = grad_input.view(grad_shape).permute(0,3,1,2).contiguous()
    return grad_input, grad_weight, grad_bias

#%%
'''
We make custom relu layer.
ReLU can replace nn.ReLU.
'''
class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        
    def forward(self, input):
        return relu.apply(input)

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output, mask, size = make_relu.make_relu_forward(input)
        ctx.save_for_backward(mask)
        ctx.input_size = size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, =  ctx.saved_tensors
        grad_input = make_relu.make_relu_backward(grad_output, mask, ctx.input_size)
        return grad_input

#%%
'''
The torch.utils.checkpoint for gradinet checkpointing is slow in ResNet.
Therfore, we make custom gradient checkpoint layer
:BnReLUConv, BnReLUConvBn, ConvBn_ARA
It only stores inpuy activation of layer and do recomputation during backward propagation
''' 
class BnReLUConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats=True, bias=False):
        super(BnReLUConv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn_weight = nn.Parameter(torch.ones(in_channels))
        self.bn_bias = nn.Parameter(torch.zeros(in_channels))
        self.register_buffer('running_mean', torch.zeros(in_channels))
        self.register_buffer('running_var', torch.zeros(in_channels))
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, input):
        return bnreluconv.apply(input, self.weight, self.bias, self.bn_weight, self.bn_bias, self.running_mean, self.running_var
                                ,self.stride, self.padding, self.groups, self.momentum, self.eps, self.track_running_stats, self.training) 
        

class bnreluconv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bn_weight, bn_bias, running_mean, running_var,
                stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats = True, training = True):
        ############# Doing batchnorm ##################
        #out, input, mu, weight_div_sqrt, sqrt = batchnorm_forward(input, bn_weight, bn_bias, training, track_running_stats, running_mean, running_var, momentum, eps)    
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean, running_var, training, momentum, eps)
        
        ############# Doing ReLU ##################        
        out = out.clamp(min=0)
        
        ############# Doing Conv2d_AS ##################
        out = F.conv2d(out, weight, bias, stride, padding, 1, groups)

        ############# Save for backward ##################
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.save_for_backward(input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias)  
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        ############# Load saved tensors & recomputation ##################
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        training = ctx.training
        momentum = ctx.momentum
        eps = ctx.eps
        
        input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reversedspace, weight, bias = ctx.saved_tensors
        running_mean_d = torch.zeros_like(running_mean).to(input.device)
        running_var_d = torch.zeros_like(running_var).to(input.device)
 
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean_d, running_var_d, training, momentum, eps)
        out = out.clamp(min=0)
        
        ############# Doing Conv2d_AS Backward ##################
        grad_weight = cudnn_convolution.convolution_backward_weight(out, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(out.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False, False)
         
        ############# Doing ReLU Backward ##################
        grad_input = grad_input.clone()
        grad_input[out <= 0] = 0
        
        ############# Doing batchnorm ##################
        grad_input, grad_bn_weight, grad_bn_bias = cudnn_batch_norm.batch_norm_backward(input, grad_input, bn_weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
       
        return grad_input, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, None, None, None, None,None, None, None, None, None        


class BnReLUConvBn(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats=True, bias=False):
        super(BnReLUConvBn, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn_weight = nn.Parameter(torch.ones(in_channels))
        self.bn_bias = nn.Parameter(torch.zeros(in_channels))
        self.register_buffer('running_mean', torch.zeros(in_channels))
        self.register_buffer('running_var', torch.zeros(in_channels))
        self.obn_weight = nn.Parameter(torch.ones(out_channels))
        self.obn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean_o', torch.zeros(out_channels))
        self.register_buffer('running_var_o', torch.zeros(out_channels))
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input):
        return bnreluconvbn.apply(input, self.weight, self.bias
                                  ,self.bn_weight, self.bn_bias, self.obn_weight, self.obn_bias
                                  ,self.running_mean, self.running_var, self.running_mean_o, self.running_var_o
                                ,self.stride, self.padding, self.groups, self.momentum, self.eps, self.track_running_stats, self.training) 


class bnreluconvbn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bn_weight, bn_bias, obn_weight, obn_bias, running_mean, running_var,  running_mean_o, running_var_o,
                stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats = True, training = True):
        ############# Doing batchnorm ##################
        out_b, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean, running_var, training, momentum, eps)
        
        ############# Doing ReLU ##################        
        out_r = out_b.clamp(min=0)
        
        ############# Doing Conv2d_AS ##################
        out_bb = F.conv2d(out_r, weight, bias, stride, padding, 1, groups)
        
        ############# Doing out batchnorm ##################
        out, save_mean_o, save_var_o, reservedspace_o = cudnn_batch_norm.batch_norm(out_bb, obn_weight, obn_bias, running_mean_o, running_var_o, training, momentum, eps)
        
        ############# Save for backward ##################
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.training = training
        ctx.track_running_statas = track_running_stats
        ctx.momentum = momentum
        ctx.eps = eps
        
        ctx.save_for_backward(input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace,
                                                                 # save for relu : None (do recomputation by input)
                              weight, bias,  # save for conv_as : mainly None(do recomputation by input) 
                              obn_weight, obn_bias, running_mean_o, running_var_o, save_mean_o, save_var_o, reservedspace_o,
                              )  # mainly None(do recomputation by input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        ############# Load saved tensors & recomputation ##################
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        training = ctx.training
        momentum = ctx.momentum
        eps = ctx.eps
        
        input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias, obn_weight, obn_bias, running_mean_o, running_var_o, save_mean_o, save_var_o, reservedspace_o = ctx.saved_tensors
        running_mean_d = torch.zeros_like(running_mean).to(input.device)
        running_var_d = torch.zeros_like(running_var).to(input.device)
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean_d, running_var_d, training, momentum, eps)
        out_ = out.clamp(min=0)
        out_b = F.conv2d(out_, weight, bias, stride, padding, 1, groups)
        
        ############# Doing batchnorm ##################
        grad_input, grad_obn_weight, grad_obn_bias = cudnn_batch_norm.batch_norm_backward(out_b, grad_output, obn_weight, running_mean_o, running_var_o, save_mean_o, save_var_o, eps, reservedspace_o)
        
        ############# Doing Conv2d_AS Backward ##################
        grad_weight = cudnn_convolution.convolution_backward_weight(out_, weight.shape, grad_input, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_input.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(out_.shape, weight, grad_input, stride, padding, (1, 1), groups, False, False, False)
            
        ############# Doing ReLU Backward ##################
        grad_input = grad_input.clone()
        grad_input[out <= 0] = 0
        
        ############# Doing batchnorm ##################
        grad_input, grad_bn_weight, grad_bn_bias = cudnn_batch_norm.batch_norm_backward(input, grad_input, bn_weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
        
        return grad_input, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, grad_obn_weight, grad_obn_bias, None, None, None, None,None, None, None, None, None, None, None , None

class ConvBn_ARA(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats=True, bias=False):
        super(ConvBn_ARA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input, ARA):
        return convbn_ara.apply(input, ARA, self.weight, self.bias, self.bn_weight, self.bn_bias, self.running_mean, self.running_var
                                ,self.stride, self.padding, self.groups, self.momentum, self.eps, self.track_running_stats, self.training) 

class convbn_ara(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ARA, weight, bias, bn_weight, bn_bias, running_mean, running_var,
                stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats = True, training = True):
        ############# Doing Conv2d ##################
        out = F.conv2d(input, weight, bias, stride, padding, 1, groups)
        
        ############# Doing batchnorm ##################
        #out, input, mu, weight_div_sqrt, sqrt = batchnorm_forward(out, bn_weight, bn_bias, training, track_running_stats, running_mean, running_var, momentum, eps)    
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(out, bn_weight, bn_bias, running_mean, running_var, training, momentum, eps)
        
        ############# Save for backward ##################
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.eps =eps
        
        ctx.save_for_backward(bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias)  
        return out, ARA.clone()
    
    @staticmethod
    def backward(ctx, grad_output, ARA):
        ############# Load saved tensors & recomputation ##################
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        eps = ctx.eps
        bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias = ctx.saved_tensors
        
        out = F.conv2d(ARA, weight, bias, stride, padding, 1, groups)
        ############# Doing batchnorm ##################
        grad_input, grad_bn_weight, grad_bn_bias = cudnn_batch_norm.batch_norm_backward(out, grad_output, bn_weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
        ############# Doing Conv2d_AS Backward ##################
        grad_weight = cudnn_convolution.convolution_backward_weight(ARA, weight.shape, grad_input, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_input.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(ARA.shape, weight, grad_input, stride, padding, (1, 1), groups, False, False, False)
        
        return grad_input, ARA, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, None, None, None, None,None, None, None, None, None
    

