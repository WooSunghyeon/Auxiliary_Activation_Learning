import torch
import torch.nn as nn
from actnn.ops import quantize_activation
from actnn.qscheme import QScheme

import aal.cudnn_conv as cudnn_convolution
import math
import torch.nn.functional as F
from actnn.ops import dequantize_activation

class Conv2d_ARA_Actnn(nn.Conv2d):
    '''
    Conv2d_ARA uses auxilairy reidual activation (ARA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    In this layer, we do not ARA to input because it has been already added outside of the module in the resnet.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, get_li=False):
        super(Conv2d_ARA_Actnn, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux):
        return conv2d_ara_actnn.apply(input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux, self.weight, self.bias, self.stride, self.padding, self.groups) 
    
class conv2d_ara_actnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux, weight, bias, stride=1, padding=0, groups=1):
        output = F.conv2d(input, weight, bias, stride, padding, 1, groups)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.saved =weight, bias
        ctx.q_shape_aux = input.shape
        return output, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux
    
    @staticmethod
    def backward(ctx, grad_output, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        weight, bias = ctx.saved
        del ctx.saved
        quantized = q_input_aux, q_bits_aux, q_scale_aux, q_min_aux
        ARA = dequantize_activation(quantized, ctx.q_shape_aux)
        del quantized
        grad_weight = cudnn_convolution.convolution_backward_weight(ARA, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False, False)
        del ARA
        grad_input = cudnn_convolution.convolution_backward_input(ctx.q_shape_aux, weight, grad_output, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        return grad_input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux, grad_weight, grad_bias, None, None, None


class distribute_ara_actnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux, scheme):
        q_input, q_bits, q_scale, q_min = quantize_activation(input, scheme)
        ctx.q_shape_aux = input.shape
        ctx.saved = q_input_aux, q_bits_aux, q_scale_aux, q_min_aux
        
        return input.clone(), q_input, q_bits, q_scale, q_min
    
    @staticmethod
    def backward(ctx, grad_output, q_input, q_bits, q_scale, q_min):
        q_input_aux, q_bits_aux, q_scale_aux, q_min_aux = ctx.saved
        del ctx.saved
        return grad_output.clone(),q_input_aux, q_bits_aux, q_scale_aux, q_min_aux, None
    
class Distribute_ARA_Actnn(nn.Module):
    '''
    When using stride to apply ARA, this layer is needed.
    If you deploy with x-> ARA_Conv2d - ARA_Conv2d - ARA_Conv2d -> y -> Distribute_ARA(y, x),
    the following three ARA_Conv2d will be trained using x.
    In this case, fisrt ARA_Conv2d act like backpropagation because it uses x which is input activation of itself. 
    '''
    def __init__(self, ):
        super(Distribute_ARA_Actnn, self).__init__()
        self.scheme = QScheme(self, num_locations=9, group=0, depthwise_groups=1, perlayer=False)
        
    def forward(self, input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux):
        return distribute_ara_actnn.apply(input, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux, self.scheme)


