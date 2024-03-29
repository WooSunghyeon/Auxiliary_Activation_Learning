# Learning-with-Auxiliary-Activation-for-Memory-Efficient-Training

This repository is the official implementation of https://openreview.net/forum?id=YgC62m4CY3r.

+ The proposed learning rule store auxiliary activations instead of actual input activations during forward propagation.
+ The proposed learning rule reduces training memory requirements without training speed reduction while achieving high performance close to backpropagation.

## Abstract

While deep learning has achieved great success in various fields, it necessitates a large amount of memory to train deep neural networks, which hinders developing massive state-of-the-art models. The reason is the conventional learning rule, backpropagation, should temporarily store input activations of all the layers in the network. To overcome this, recent studies suggested various memory-efficient implementations of backpropagation. However, those approaches incur computational overhead due to the recomputation of activations, slowing down neural network training. In this work, we propose a new learning rule which significantly reduces memory requirements while closely matching the performance of backpropagation. The algorithm combines auxiliary activation with output activation during forward propagation, while only auxiliary activation is used during backward propagation instead of actual input activation to reduce the amount of data to be temporarily stored. We mathematically show that our learning rule can reliably train the networks whose loss landscape is convex if the auxiliary activation satisfies certain conditions. Based on this observation, we suggest candidates of auxiliary activation that satisfy those conditions. Experimental results confirm that the proposed learning rule achieves competitive performance compared to backpropagation in various models such as ResNet, Transformer, BERT, ViT, and MLP-Mixer.

![gitfig](https://user-images.githubusercontent.com/114454500/192741100-52b870ac-21e0-40de-bac6-2a9c11bfae1d.png)

## Install

+ Requirements
```bash
conda env create -f aal.yaml
conda activate aal
```

+ Buld AAL:
```bash
cd setpup/aal
pip install -v -e .
```

We also slightly modified code of ActNN and Mesa to apply it with aal simultaneously.

+ Buld ActNN for AAL
```bash
conda activate aal
cd setpup/actnn
pip install -v -e .
```
+ Buld Mesa for AAL
```bash
conda activate aal
cd setpup/mesa
python setup.py develop
```

## Usage 

+ Implementing ARA
```python
from aal.aal import Conv2d_ARA, Distribute_ARA

# define convolution layer wich uses ARA
self.conv1 = nn.Conv2d(64,64,3,1,1)
self.conv2 = Conv2d_ARA(64,64,3,1,1)
self.conv3 = Conv2d_ARA(64,64,3,1,1)
# define Distribute_ARA which is layer for implementing ARA
self.dsa = Distribute_ARA()

# define auxiliary residual activation for updating Conv2d_ARA
ARA = x.clone()
# doing backpropagation for conv1
x = self.conv1(x)
# adding auxiliary activation to output activation (residual connection)
# and propagating to Conv2d_ARA
x += ARA
x, ARA = self.conv2(x, ARA)
X += ARA
x, ARA = self.conv3(x, ARA)
# Distribute ARA makes self.conv2 and self.conv3 updates weight with ARA, not x!
x, ARA = self.dsa(x, ARA)
```

+ Implementing ARA+ActNN
```python
import actnn
from actnn.aal_actnn import Conv2d_ARA_Actnn, Distribute_ARA_Actnn
from actnn.ops import quantize_activation
from actnn.qscheme import QScheme

# define convolution layer wich uses ARA_Actnn
self.conv1 = actnn.Conv2d(64,64,3,1,1)
self.conv2 = Conv2d_ARA_Actnn(64,64,3,1,1)
self.conv3 = Conv2d_ARA_Actnn(64,64,3,1,1)
# define Distribute_ARA_Actnn which is layer for implementing ARA_Actnn
self.dsa = Distribute_ARA_Actnn()

# define auxiliary residual activation for updating Conv2d_ARA_actnn
ARA = x.clone()

# define quantized auxiliary residual activation
# In Conv2d_ARA_Actnn, we use q_ara, ... instead of ARA for memory optimization when performing backpropagation
q_ara, q_ara_bits, q_ara_scale, q_ara_min = quantize_activation(ARA, self.scheme)

# doing backpropagation for conv1
x = self.conv1(x)
# adding auxiliary activation to output activation (residual connection)
# and propagating q_ara to Conv2d_ARA_Actnn
x += ARA
x, q_ara, q_ara_bits, q_ara_scale, q_ara_min = self.conv2(x, q_ara, q_ara_bits, q_ara_scale, q_ara_min)
X += ARA
x, q_ara, q_ara_bits, q_ara_scale, q_ara_min = self.conv3(x, q_ara, q_ara_bits, q_ara_scale, q_ara_min)
# Distribute q_ara makes self.conv2 and self.conv3 updates weight with q_ara, not x!
x, q_ara, q_ara_bits, q_ara_scale, q_ara_min = self.dsa(x, q_ara, q_ara_bits, q_ara_scale, q_ara_min)
```


+ Implementing ASA
```python
from aal.aal import Linear_ASA

# define linear layer for ASA
self.linear = Linear_asa(256,256)

# propagating to Linear_ASA layer
# it would perform add 1-bit auxiliary sign activation during forward propagation
# and store this 1-bit auxiliary sign activation.
# during backprop, it would update weights by 1-bit auxiliary sign activation
x = self.linear(x)
```

## Example

[ResNet](https://github.com/asdfasgqergadsad/Auxiliary_Activation_Learning/tree/main/experiments/ResNet)

[Transformer](https://github.com/asdfasgqergadsad/Auxiliary_Activation_Learning/tree/main/experiments/Transformer)

[BERT_L](https://github.com/asdfasgqergadsad/Auxiliary_Activation_Learning/tree/main/experiments/BERT_L)

[ViT_L](https://github.com/asdfasgqergadsad/Auxiliary_Activation_Learning/tree/main/experiments/ViT_L)

[MLP-Mixer_L](https://github.com/asdfasgqergadsad/Auxiliary_Activation_Learning/tree/main/experiments/MLP-Mixer_L)


## Results

![RESULTS_GOOD](https://github.com/WooSunghyeon/Auxiliary_Activation_Learning/assets/85105077/7dd7bf77-79a0-434e-9e0f-ae4ee9a41420)

![result2](https://user-images.githubusercontent.com/114454500/192739585-98379113-d735-47e9-b72a-84e92028b3b3.png)

## Acknowledgments
  
  In this repository, code of [ActNN](https://github.com/ucbrise/actnn) and [Mesa](https://github.com/ziplab/Mesa) are modified to apply with our AAL.
  Thanks the authors for open-source code.
  
 ## Lisense

> All content in this repository is licensed under the MIT license. 

