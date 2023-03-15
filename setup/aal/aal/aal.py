import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import aal.cudnn_conv as cudnn_convolution
import aal.make_asa as make_asa


#%%
def learning_indicator(real_act, alternative_act, convnet=False):
    '''
    This function can calculate learning_indicator by using real_act and alternative_act. 
    '''
    if convnet:
        bp_direction = real_act*real_act
        as_direction = alternative_act * (2*real_act - alternative_act)
        bp_direction = torch.sum(bp_direction.view(bp_direction.size(0),-1), dim=1) + 1e-6
        as_direction = torch.sum(as_direction.view(as_direction.size(0),-1), dim=1)
        li = torch.flatten(as_direction/bp_direction).view(1,-1)
        li.flatten()[:50].detach().clone()
    else:
        bp_direction = torch.sum(real_act*real_act, dim=2) + 1e-6
        as_direction = torch.sum(alternative_act * (2*real_act - alternative_act), dim=2)
        li = torch.flatten(as_direction/bp_direction).tolist()
    return li


#%%
class Conv2d_ARA(nn.Conv2d):
    '''
    Conv2d_ARA uses auxilairy reidual activation (ARA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    In this layer, we do not ARA to input because it has been already added outside of the module in the resnet.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, get_li=False):
        super(Conv2d_ARA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.li=None
        self.get_li = get_li
        
    def forward(self, input, ARA):
        if self.get_li and self.training:
            self.li = learning_indicator(input, ARA, convnet=True)
        return conv2d_ara.apply(input, ARA, self.weight, self.bias, self.stride, self.padding, self.groups) 
    
class conv2d_ara(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ARA, weight, bias, stride=1, padding=0, groups=1):
        output = F.conv2d(input, weight, bias, stride, padding, 1, groups)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.saved =weight, bias
        return output, ARA.clone()
    
    @staticmethod
    def backward(ctx, grad_output, ARA):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        weight, bias = ctx.saved
        del ctx.saved
        grad_weight = cudnn_convolution.convolution_backward_weight(ARA, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False, False)
        grad_input = cudnn_convolution.convolution_backward_input(ARA.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        return grad_input, ARA, grad_weight, grad_bias, None, None, None, None,None
    
class Distribute_ARA(nn.Module):
    '''
    When using stride to apply ARA, this layer is needed.
    If you deploy with x-> ARA_Conv2d - ARA_Conv2d - ARA_Conv2d -> y -> Distribute_ARA(y, x),
    the following three ARA_Conv2d will be trained using x.
    In this case, fisrt ARA_Conv2d act like backpropagation because it uses x which is input activation of itself. 
    '''
    def __init__(self, ):
        super(Distribute_ARA, self).__init__()
        
    def forward(self, input, auxiliary_activation):
        return distribute_ara.apply(input, auxiliary_activation)

class distribute_ara(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, auxiliary_activation):
        output = input.clone()
        ctx.saved = auxiliary_activation
        #torch.cuda.empty_cache()

        return output, output.detach()
    
    @staticmethod
    def backward(ctx, grad_output, auxiliary_activation):
        auxiliary_activation = ctx.saved
        del ctx.saved
        #torch.cuda.empty_cache()
        return grad_output.clone(), auxiliary_activation.clone()



#%%
class Linear_ASA(nn.Linear):
    '''
    Linear_ABA uses auxiliary sign activation (ASA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    You have to use relu=True if Linear_ASA is displayed after ReLU function.
    The epsilon controls the magnitude of ARA to make comparable value to output activation.
    The learning rate can be enlarged by setting lr_expansion.
    '''
    def __init__(self, in_features, out_features, bias=True, get_li=False, relu=False, epsilon = 0.01, lr_expansion=100):
        super(Linear_ASA, self).__init__(in_features, out_features, bias=bias)
        self.get_li = get_li
        self.epsilon = epsilon
        self.relu = relu # if relu is True, dequantization makes value 1,0. else, dequantization makes value 1,-1. 
        self.lr_expansion = lr_expansion
        
    def forward(self, input):
        if self.get_li and self.training:
            ASA = input.sign().detach().clone()
            self.li = learning_indicator(input+ASA*self.epsilon, ASA*self.epsilon)
        return linear_asa.apply(input, self.weight, self.bias, self.relu, self.epsilon, self.lr_expansion)
        
class linear_asa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, relu=False, epsilon=0.01, lr_expansion=100):
        ASA, mask, size = make_asa.make_asa_forward(input, relu)
        output = F.linear(input + ASA*epsilon, weight, bias)
        ctx.input_size = input.size()
        ctx.relu=relu
        ctx.epsilon=epsilon
        ctx.lr_expansion=lr_expansion
        ctx.saved = mask, bias, weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ASA, bias, weight = ctx.saved
        del ctx.saved
        ASA = make_asa.make_asa_backward(grad_output, ASA, ctx.input_size, ctx.relu)
        
        grad_output_size = grad_output.size()
        grad_output = grad_output.reshape(-1, grad_output_size[-1]) 
        ASA = ASA.reshape(-1,ASA.size(2))
        grad_weight = F.linear(ASA.t()*ctx.epsilon, grad_output.t()).t()
        grad_weight *= ctx.lr_expansion
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight.t())
        return grad_input.reshape(ctx.input_size), grad_weight, grad_bias, None, None, None
    
class Conv2d_ASA(nn.Conv2d):
    '''
    Conv2d_ABA uses auxiliary sign activation (ASA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    The epsilon controls the magnitude of ARA to make comparable value to output activation.
    The learning rate can be enlarged by setting lr_expansion.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, relu=False, epsilon = 0.01, lr_expansion=100, get_li=False):
        super(Conv2d_ASA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.epsilon = epsilon
        self.lr_expansion=lr_expansion
        self.li=None
        self.get_li=get_li
        self.relu = relu # if relu is True, dequantization makes value 1,0. else, dequantization makes value 1,-1.
        
    def forward(self, input):
        ASA = input.sign().detach().clone()
        if self.get_li and self.training:
            self.li = learning_indicator(input+self.epsilon*ASA, self.epsilon*ASA, convnet=True)
        return conv2d_asa.apply(input, ASA, self.weight, self.bias, self.stride, self.padding, self.groups, self.epsilon, self.lr_expansion, self.relu) 

class conv2d_asa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ASA, weight, bias, stride=1, padding=0, groups=1, epsilon=0.01, lr_expansion=100, relu=False):
        output = F.conv2d(input+ASA*epsilon, weight, bias, stride, padding, 1, groups)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        #packbits
        ASA, mask, size = make_asa.make_asa_forward(input, relu)
        ctx.size = size
        ctx.epsilon=epsilon
        ctx.lr_expansion=lr_expansion
        ctx.saved = ASA, mask, weight, bias
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        ASA, mask, weight, bias = ctx.saved
        del ctx.saved
        #unpackbit
        ASA = make_asa.make_asa_backward(grad_output, ASA, ctx.input_size, ctx.relu)
        grad_weight = cudnn_convolution.convolution_backward_weight(ASA*ctx.epsilon, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False,False)
        grad_weight *= ctx.lr_expansion
        grad_input = cudnn_convolution.convolution_backward_input(ASA.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False,False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        return grad_input, None, grad_weight, grad_bias, None, None, None, None,None, None, None, None


