import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from aal.aal import Conv2d_ARA, Conv2d_ASA, Distribute_ARA
from actnn.aal_actnn import Conv2d_ARA_Actnn, Distribute_ARA_Actnn
from aal.custom_layer import BatchNorm2d, ReLU, BnReLUConv, BnReLUConvBn, ConvBn_ARA
from actnn.ops import quantize_activation
from actnn.qscheme import QScheme

class ResNet(nn.Module):
    def __init__(self, dataset='cifar10', model = 'resnet18'):
        super(ResNet, self).__init__()
        block = BasicBlock
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny-imagenet':    
            num_classes = 200
        elif dataset == 'imagenet':    
            num_classes = 1000
        self.dataset = dataset
        self.num_classes = num_classes
        
        # Decide number of layer for resnet
        if model == 'resnet18':
            num_blocks = [2,2,2,2]
        elif model == 'resnet34':
            num_blocks = [3,4,6,3]
        elif model == 'resnet50':
            num_blocks = [3,4,6,3]
            block = Bottleneck
        elif model == 'resnet101':
            num_blocks = [3,4,23,3]
            block = Bottleneck
        elif model == 'resnet152':
            num_blocks = [3,8,36,3]
            block = Bottleneck
           
        self.conv0 = nn.Conv2d(3,64,3,1,1, bias = False)
        if dataset == 'tiny-imagenet':
            self.conv0 = nn.Conv2d(3, 64,3,2,1, bias = False)
        elif dataset == 'imagenet':
            self.conv0 = nn.Conv2d(3,64,7,2,3, bias = False)
            self.maxpool = nn.MaxPool2d(3,2,1)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = ReLU()
        
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Sequential(
                            nn.Linear(512*block.expansion, num_classes),
                            )
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.relu(self.bn0(self.conv0(x)))
        if self.dataset == 'imagenet':
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out, []
    
class ResNet_ARA(nn.Module):
    def __init__(self, dataset='cifar10', model = 'resnet18', ARA_stride = [2,2,2,2], actnn=False):
        super(ResNet_ARA, self).__init__()
        self.model = model
        self.dataset = dataset
        self.actnn = actnn
        self.scheme = QScheme(self, num_locations=9, group=0, depthwise_groups=1, perlayer=False)
        
        block = ARA_BasicBlock
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny-imagenet':    
            num_classes = 200
        elif dataset == 'imagenet':    
            num_classes = 1000
       
        # Decide number of layer for resnet 
        # While Basic Block includes stride =2 first order,
        # ASBasick block includes stride = 2 last order
        # so num_blocks =[4,4,6,2] is equal to [3,4,6,3] in resnet for ResNet-34
        if model == 'resnet18':
            num_blocks = [3,2,2,1]
        elif model == 'resnet34':
            num_blocks = [4,4,6,2]
        elif model == 'resnet50':
            num_blocks = [4,4,6,2]
            block = ARA_Bottleneck
        elif model == 'resnet101':
            num_blocks = [4,4,23,2]
            block = ARA_Bottleneck
        elif model == 'resnet152':
            num_blocks = [4,8,36,2]
            block = ARA_Bottleneck
                        
        # Make architecture
        self.conv0 = nn.Conv2d(3,64,3,1,1, bias = False)
        if dataset == 'tiny-imagenet':
            self.conv0 = nn.Conv2d(3, 64,3,2,1, bias = False)
        elif dataset == 'imagenet':
            self.conv0 = nn.Conv2d(3,64,7,2,3, bias = False)
            self.maxpool = nn.MaxPool2d(3,2,1)
        self.bn0 = nn.BatchNorm2d(64) 
        self.relu = ReLU()
        
        self.in_planes = 64*block.expansion
        
        if model == 'resnet18' or model == 'resnet34':
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, k = ARA_stride[0], actnn=actnn)
        else:
            self.layer0 = Bottleneck(64, 64, 1)
            # self.layer0 is first block with 64 channel dimensions.
            # Although it is ARA_Bottleneck_GCP, they do backpropagation because ARA_Conv uses actual input activation as out=ARA_Conv(x, ARA) and x=ARA.
            # covbn = True means the gradient checkpoiting is applied to shortcut (conv-bn)
            # only self.layer0 is convbn=True because the shortcut in self.layer0 consumes lots of memory
            self.layer1 = self._make_layer(block, 64, num_blocks[0]-1, stride=1, k = ARA_stride[0], actnn=actnn)
            
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, k = ARA_stride[1], actnn=actnn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1, k = ARA_stride[2], actnn=actnn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, k = ARA_stride[3], actnn=actnn)
        self.classifier = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(512*block.expansion, num_classes)
                                        )
        
        
    def _make_layer(self, block, planes, num_blocks, stride, k = 2, actnn=False):
        strides = [1]*(num_blocks-1) + [2] # ASBasick block includes stride = 2 last order
        layers = []
        for i, stride in enumerate(strides):
            if (i) % k == k-1 or stride == 2:
                distribute = True    
            else:
                distribute = False
                
            if i != num_blocks-1:
                layers.append(block(self.in_planes, planes, stride, distribute = distribute, actnn=actnn))
                self.in_planes = planes * block.expansion
            else:
                if planes != 512:
                    layers.append(block(self.in_planes, 2 * planes, stride, distribute = distribute, actnn=actnn))
                    self.in_planes = 2 * planes * block.expansion
                else:
                    layers.append(block(self.in_planes, planes, 1, distribute = distribute, actnn=actnn))
                    self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if not(self.actnn):
            x = self.relu(self.bn0(self.conv0(x)))
            if self.dataset == 'imagenet':
                x = self.maxpool(x)
            ARA = x.detach().clone()
            if self.model == 'resnet18' or self.model == 'resnet34':
                x = x, ARA
                out = self.layer1(x)
            else:
                out = self.layer0(x)
                ARA = out.detach().clone()
                out = out, ARA
                out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out, ARA = self.layer4(out)
            out = self.classifier(out)
        else:
            x = self.relu(self.bn0(self.conv0(x)))
            if self.dataset == 'imagenet':
                x = self.maxpool(x)
            ARA = x.detach().clone()
            if self.model == 'resnet18' or self.model == 'resnet34':
                x = x, ARA
                out = self.layer1(x)
            else:
                out = self.layer0(x)
                ARA = out.detach().clone()
                q_input, q_bits, q_scale, q_min = quantize_activation(ARA, self.scheme)
                out = out, q_input, q_bits, q_scale, q_min
                out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out, q_input, q_bits, q_scale, q_min = self.layer4(out)
            out = self.classifier(out)
            
        return out, []
    
    
class ResNet_ARA_GCP(nn.Module):
    def __init__(self, dataset='cifar10', model = 'resnet18', ARA_stride = [2,2,2,2], get_li=False):
        super(ResNet_ARA_GCP, self).__init__()
        self.model = model
        self.dataset = dataset
        block = ARA_BasicBlock_GCP
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny-imagenet':    
            num_classes = 200
        elif dataset == 'imagenet':    
            num_classes = 1000
    
        # Decide number of layer for resnet 
        # While Basic Block includes stride =2 first order,
        # RA_Basick block includes stride = 2 last order
        # so num_blocks =[4,4,6,2] is equal to [3,4,6,3] in resnet for ResNet-34
        if model == 'resnet18':
            num_blocks = [3,2,2,1]
        elif model == 'resnet34':
            num_blocks = [4,4,6,2]
        elif model == 'resnet50':
            num_blocks = [4,4,6,2]
            block = ARA_Bottleneck_GCP
        elif model == 'resnet101':
            num_blocks = [4,4,23,2]
            block = ARA_Bottleneck_GCP
        elif model == 'resnet152':
            num_blocks = [4,8,36,2]
            block = ARA_Bottleneck_GCP
                        
        # Make architecture
        self.v1 = V1_filter(dataset=dataset)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.in_planes = 64*block.expansion
        self.get_li = get_li
        if model == 'resnet18' or model == 'resnet34':
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, k = ARA_stride[0], get_li=self.get_li)
        else:
            self.layer0 = nn.Sequential(ARA_Bottleneck_GCP(64, 64, 1 , True, convbn = True)) 
            # self.layer0 is first block with 64 channel dimensions.
            # Although it is ARA_Bottleneck_GCP, they do backpropagation because ARA_Conv uses actual input activation as out=ARA_Conv(x, ARA) and x=ARA.
            # covbn = True means the gradient checkpoiting is applied to shortcut (conv-bn)
            # only self.layer0 is convbn=True because the shortcut in self.layer0 consumes lots of memory.
            self.layer1 = self._make_layer(block, 64, num_blocks[0]-1, stride=1, k = ARA_stride[0], get_li=self.get_li)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, k = ARA_stride[1], get_li=self.get_li)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1, k = ARA_stride[2], get_li=self.get_li)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, k = ARA_stride[3], get_li=self.get_li)
        self.classifier = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(512*block.expansion, num_classes)
                                        )
         
    def _make_layer(self, block, planes, num_blocks, stride, k = 2, get_li=False):
        strides = [1]*(num_blocks-1) + [2] # RA_Basick block includes stride = 2 last order
        layers = []
        for i, stride in enumerate(strides):
            if (i) % k == k-1 or stride == 2:
                distribute = True    
            else:
                distribute = False
            if get_li:
                if (i) % k == 0:
                    get_li_=False    
                else:
                    get_li_=get_li
            else:
                get_li_=False
                    
            if i != num_blocks-1:
                layers.append(block(self.in_planes, planes, stride, distribute = distribute, get_li=get_li_))
                self.in_planes = planes * block.expansion            
            else:
                if planes != 512:
                    layers.append(block(self.in_planes, 2 * planes, stride, distribute = distribute, get_li=get_li_))
                    self.in_planes = 2 * planes * block.expansion
                else:
                    layers.append(block(self.in_planes, planes, 1, distribute = distribute, get_li=get_li_))
                    self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = checkpoint(self.v1, x, self.dummy_tensor)
        #x = self.v1(x, self.dummy_tensor)
        ARA = x.detach().clone()
        x = x,ARA
        if self.model == 'resnet18' or self.model == 'resnet34':
            out = self.layer1(x)
        else:
            out = self.layer0(x)
            out = self.layer1(out)            
        out = self.layer2(out)
        out = self.layer3(out)
        out, ARA = self.layer4(out)
        out = self.classifier(out)
        
        li=None
        if self.get_li and self.training:
            for m in self.modules():
              if isinstance(m, Conv2d_ARA):  
                  if li==None:
                      li = m.li
                  else:
                      if m.li != None:
                          li = torch.cat((li,m.li)).detach().clone()           
        return out, li

class ResNet_ASA_GCP(nn.Module):
    def __init__(self, dataset='cifar10', model = 'resnet18', learning_rule='asa', ARA_stride = [2,2,2,2], get_li=False):
        super(ResNet_ASA_GCP, self).__init__()
        self.model = model
        self.dataset = dataset
        block = ASA_BasicBlock
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny-imagenet':    
            num_classes = 200
        elif dataset == 'imagenet':    
            num_classes = 1000
       
        # Decide number of layer for resnet 
        # While Basic Block includes stride =2 first order,
        # RA_Basick block includes stride = 2 last order
        # so num_blocks =[4,4,6,2] is equal to [3,4,6,3] in resnet for ResNet-34
        if model == 'resnet18':
            num_blocks = [3,2,2,1]
        elif model == 'resnet34':
            num_blocks = [4,4,6,2]
        elif model == 'resnet50':
            num_blocks = [4,4,6,2]
            block = ASA_Bottleneck
        elif model == 'resnet101':
            num_blocks = [4,4,23,2]
            block = ASA_Bottleneck
        elif model == 'resnet152':
            num_blocks = [4,8,36,2]
            block = ASA_Bottleneck
            
        # Make architecture
        self.v1 = V1_filter(dataset=dataset)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.in_planes = 64*block.expansion
        self.get_li = get_li
        
        if model == 'resnet18' or model == 'resnet34':
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, k = ARA_stride[0],learning_rule=learning_rule, get_li=get_li)
        else:
            self.layer0 = nn.Sequential(ARA_Bottleneck_GCP(64, 64, 1 , True, convbn = True))
            # self.layer0 is first block with 64 channel dimensions.
            # Although it is ARA_Bottleneck_GCP, they do backpropagation because ARA_Conv uses actual input activation as out=ARA_Conv(x, ASA) and x=ASA.
            # covbn = True means the gradient checkpoiting is applied to shortcut (conv-bn)
            # only self.layer0 is convbn=True because the shortcut in self.layer0 consumes lots of memory.
            
            self.layer1 = self._make_layer(block, 64, num_blocks[0]-1, stride=1, k = ARA_stride[0], learning_rule=learning_rule, get_li=get_li)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, k = ARA_stride[1], learning_rule=learning_rule, get_li=get_li)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1, k = ARA_stride[2], learning_rule=learning_rule, get_li=get_li)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, k = ARA_stride[3], learning_rule=learning_rule, get_li=get_li)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(512*block.expansion, num_classes)
                                        )
         
    def _make_layer(self, block, planes, num_blocks, stride, k = 2, learning_rule = 'asa', get_li=False):
        strides = [1]*(num_blocks-1) + [2] # ASA_Basick block includes stride = 2 last order
        layers = []
        for i, stride in enumerate(strides):
            if (i) % k == 0:
                bp_conv = True    
            else:
                bp_conv = False
            if i != num_blocks-1:
                layers.append(block(self.in_planes, planes, stride, learning_rule=learning_rule, bp_conv=bp_conv, get_li=get_li))
                self.in_planes = planes * block.expansion            
            else:
                if planes != 512:
                    layers.append(block(self.in_planes, 2 * planes, stride, learning_rule=learning_rule, bp_conv=bp_conv, get_li=get_li))
                    self.in_planes = 2 * planes * block.expansion
                else:
                    layers.append(block(self.in_planes, planes, 1, learning_rule=learning_rule, bp_conv=bp_conv, get_li=get_li))
                    self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = checkpoint(self.v1, x, self.dummy_tensor)
        if self.model == 'resnet18' or self.model == 'resnet34':
            out = self.layer1(x)
        else:
            x = x, x.detach().clone()
            out, _ = self.layer0(x)    
            out = self.layer1(out)            
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classifier(out)
        
        li=None
        if self.get_li and self.training:
            for m in self.modules():
              if isinstance(m, Conv2d_ASA) or isinstance(m, Conv2d_ASA):  
                  if li==None:
                      li = m.li
                  else:
                      if m.li != None:
                          li = torch.cat((li,m.li)).detach().clone()
        return out, li
    
    
class BRC(nn.Module): 
    def __init__(self, in_planes, planes, kernel_size=1, stride=1, padding=0, bias = False):
        super(BRC, self).__init__()
        self.bnreluconv = nn.Sequential(nn.BatchNorm2d(in_planes),
                                        ReLU(),
                                        nn.Conv2d(in_planes,planes, kernel_size=kernel_size, 
                                                  stride=stride, padding=padding, bias = bias))
    def forward(self,x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.bnreluconv(x)
        return x
    
class V1_filter(nn.Module):
    def __init__(self, dataset = 'cifar10'):
        super(V1_filter, self).__init__()
        
        self.v1 = nn.Sequential(nn.Conv2d(3,64, 3, 1, 1, bias = False),
                                        nn.BatchNorm2d(64),
                                        ReLU(),
                                        )
        if dataset == 'tiny-imagenet':
            self.v1 = nn.Sequential(nn.Conv2d(3,64, 3, 2, 1, bias = False),
                                        nn.BatchNorm2d(64),
                                        ReLU(),
                                        )
        elif dataset == 'imagenet':
            self.v1 = nn.Sequential(nn.Conv2d(3,64, 7, 2, 3, bias = False),
                                        nn.BatchNorm2d(64),
                                        ReLU(),
                                        nn.MaxPool2d(3,2,1)
                                        )
            
    def forward(self,x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.v1(x)
        return x

         
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, learning_rule='bp'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu2 = ReLU()
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out    
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, mode='bp', connect_features = 10):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu1=ReLU()
        self.relu2=ReLU()
        self.relu3=ReLU()
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out    
    
class ARA_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, distribute = False, actnn = False):
        super(ARA_BasicBlock, self).__init__()
        self.conv1 = Conv2d_ARA(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if actnn:
            self.conv1 = Conv2d_ARA_Actnn(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1, padding=1, bias= False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.relu2 = ReLU()
        if stride != 1 or in_planes != self.expansion*planes:
            self.short_conv = Conv2d_ARA(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            if actnn:
                self.short_conv = Conv2d_ARA_Actnn(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False)
            self.short_bn = nn.BatchNorm2d(self.expansion*planes)
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.distribute = distribute
        if distribute:
            self.dsa = Distribute_ARA()
            if actnn:
                self.dsa = Distribute_ARA_Actnn()
                
    def forward(self, x):
        x, ARA = x
        out, ARA = self.conv1(x,ARA) # when calculates weight updates, use ARA instead of x
        out = self.relu1(self.bn1(out))
        out = self.bn2(self.conv2(out))
        if self.stride == 1 and self.in_planes == self.expansion*self.planes:
            out += x
        else:
            x_, ARA = self.short_conv(x, ARA)
            out += self.short_bn(x_) # when calculates weight updates, use ARA instead of x 
        out = self.relu2(out)
        if self.distribute:
            out, ARA = self.dsa(out, ARA)
        out = out, ARA
        return out

class ARA_Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, distribute = False, actnn = False):
        super(ARA_Bottleneck, self).__init__()
        self.conv1 = Conv2d_ARA(in_planes, planes, kernel_size=1, bias=False)
        if actnn:
            self.conv1 = Conv2d_ARA_Actnn(in_planes, planes, kernel_size=1, bias=False)
                
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3,
                               stride=stride, padding=1, bias= False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.short_conv = Conv2d_ARA(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            if actnn:
                self.short_conv = Conv2d_ARA_Actnn(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.short_bn = nn.BatchNorm2d(self.expansion*planes)
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.distribute = distribute
        if distribute:
            self.dsa = Distribute_ARA()
            if actnn:
                self.dsa = Distribute_ARA_Actnn()
    
        self.relu1=ReLU()
        self.relu2=ReLU()
        self.relu3=ReLU()
        
        self.actnn = actnn
            
    def forward(self, x):
        if not(self.actnn):
            x, ARA = x
            out, ARA = self.conv1(x,ARA) # when calculates weight updates, use ARA instead of x
            out = self.relu1(self.bn1(out))
            out = self.relu2(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.stride == 1 and self.in_planes == self.expansion*self.planes:
                out += x
            else:
                x_, ARA = self.short_conv(x, ARA)
                out += self.short_bn(x_) # when calculates weight updates, use ARA instead of x 
            out = self.relu3(out)
            if self.distribute:
                out, ARA = self.dsa(out, ARA)
            out = out, ARA
        else:
            x, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux = x
            out, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux = self.conv1(x,q_input_aux, q_bits_aux, q_scale_aux, q_min_aux) # when calculates weight updates, use ARA instead of x
            out = self.relu1(self.bn1(out))
            out = self.relu2(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.stride == 1 and self.in_planes == self.expansion*self.planes:
                out += x
            else:
                x_, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux = self.short_conv(x, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux)
                out += self.short_bn(x_) # when calculates weight updates, use ARA instead of x 
            out = self.relu3(out)
            if self.distribute:
                out, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux = self.dsa(out, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux)
            out = out, q_input_aux, q_bits_aux, q_scale_aux, q_min_aux
            
        return out

class ARA_BasicBlock_GCP(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, distribute = False, get_li=False):
        super(ARA_BasicBlock_GCP, self).__init__()
        self.conv1 = Conv2d_ARA(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, get_li=get_li)
        self.bnreluconv1 = BnReLUConv(planes, planes, kernel_size = 3, stride=1, padding=1, bias= False)
        self.bn2 = BatchNorm2d(planes)  
        if stride != 1 or in_planes != self.expansion*planes:
            self.short_conv = Conv2d_ARA(in_planes, self.expansion*planes, 
                                           kernel_size=1, stride=stride, bias=False, get_li=get_li)
            self.short_bn = BatchNorm2d(self.expansion*planes)
        self.relu3 = ReLU()
        self.in_planes = in_planes
        self.planes = planes
        self.distribute = distribute
        self.stride = stride
        if distribute:
            self.dsa = Distribute_ARA()
        
    def forward(self, x):
        x, ARA = x
        out, ARA = self.conv1(x,ARA) # when calculates weight updates, use ARA instead of x
        out = self.bnreluconv1(out)
        out = self.bn2(out)
        if self.stride == 1 and self.in_planes == self.expansion*self.planes:
            out += x
        else:
            x_, ARA = self.short_conv(x, ARA)
            out += self.short_bn(x_) # when calculates weight updates, use ARA instead of x 
        out = self.relu3(out)
        
        if self.distribute: #in the end of stride, ARA are distributed for updating weights of ARA_Conv bt stride
            out, ARA = self.dsa(out, ARA)
        out = out, ARA
        return out

    
class ARA_Bottleneck_GCP(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, distribute = False, convbn = False, get_li=False):
        super(ARA_Bottleneck_GCP, self).__init__()
        self.conv1 = Conv2d_ARA(in_planes, planes, kernel_size=1, bias=False, get_li=get_li)
        self.bnreluconv1 = BnReLUConv(planes, planes, kernel_size = 3, stride=stride, padding=1, bias= False)
        self.bnreluconvbn2 = BnReLUConvBn(planes, self.expansion * planes, kernel_size=1, stride=1, padding=0, bias = False)
        self.relu3 = ReLU()
        self.shortcut = nn.Sequential()
        if convbn:
            # it only used for first self.layer0 block in bottleneck
            # By this, we can apply gradient checkpointing to conv-bn in shortcut
            # Although it is ConvBN_ARA layer, it acts like backpropagation 
            # because it uses it's own input activation for updating weights by distribute=True 
            self.short_convbn = ConvBn_ARA(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False) 
        else:
            if stride != 1 or in_planes != self.expansion*planes:
                self.short_conv = Conv2d_ARA(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False, get_li=get_li)
                self.short_bn = nn.BatchNorm2d(self.expansion*planes)
        self.in_planes = in_planes
        self.planes = planes
        self.distribute = distribute
        if distribute:
            self.dsa = Distribute_ARA()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)    
        self.stride = stride
        self.convbn = convbn
        
    def forward(self, x):
        x, ARA = x
        out, ARA = self.conv1(x,ARA) # when calculates weight updates, use ARA instead of x
        out = self.bnreluconv1(out)
        out = self.bnreluconvbn2(out)
        if self.stride == 1 and self.in_planes == self.expansion*self.planes:
            out += x
        else:
            if self.convbn:
                x_, ARA = self.short_convbn(x, ARA)
                out += x_
            else:
                x_, ARA = self.short_conv(x, ARA) # when calculates weight updates, use ARA instead of x
                out += self.short_bn(x_)  
        out = self.relu3(out)
        
        if self.distribute: #in the end of stride, ARA are distributed for updating weights of ARA_Conv bt stride
            out, ARA = self.dsa(out, ARA)
        out = out, ARA
        return out
    
class ASA_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, learning_rule='asa', bp_conv= False, get_li=False):
        super(ASA_BasicBlock, self).__init__()
        self.conv1 = Conv2d_ASA(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, get_li=get_li)
        if bp_conv:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)    
        
        self.bnreluconv1 = BnReLUConv(planes, planes, kernel_size = 3, stride=1, padding=1, bias= False)
        self.bn2 = BatchNorm2d(planes)  
        if stride != 1 or in_planes != self.expansion*planes:
            self.short_conv = Conv2d_ASA(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, get_li=get_li)
            if bp_conv:
                self.short_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.short_bn = BatchNorm2d(self.expansion*planes)
        self.relu3 = ReLU()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
         
    def forward(self, x):
        out = self.conv1(x) 
        out = self.bnreluconv1(out)
        out = self.bn2(out)
        if self.stride == 1 and self.in_planes == self.expansion*self.planes:
            out += x
        else:
            x_ = self.short_conv(x)
            out += self.short_bn(x_) 
        out = self.relu3(out)

        return out

    
class ASA_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, learning_rule='aba', bp_conv=False, get_li=False):
        super(ASA_Bottleneck, self).__init__()
        self.conv1 = Conv2d_ASA(in_planes, planes, kernel_size=1, bias=False, get_li=get_li)
        if bp_conv:    
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bnreluconv1 = BnReLUConv(planes, planes, kernel_size = 3, stride=stride, padding=1, bias= False)
        self.bnreluconvbn2 = BnReLUConvBn(planes, self.expansion * planes, kernel_size=1, stride=1, padding=0, bias = False)
        self.relu3 = ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.short_conv = Conv2d_ASA(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, get_li=get_li)
            if bp_conv:
                self.short_conv = nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False)
            self.short_bn = nn.BatchNorm2d(self.expansion*planes)
        self.in_planes = in_planes
        self.planes = planes
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)    
        self.stride = stride
        
    def forward(self, x):
        out = self.conv1(x) 
        out = self.bnreluconv1(out)
        out = self.bnreluconvbn2(out)
        if self.stride == 1 and self.in_planes == self.expansion*self.planes:
            out += x
        else:
            x_ = self.short_conv(x)
            out += self.short_bn(x_) 
        out = self.relu3(out)
        return out
    
