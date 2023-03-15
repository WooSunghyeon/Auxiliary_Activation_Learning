# Experiments on ResNet

See help (--h flag) for available options before executing the code.

We prepaired Tiny-ImageNet and IWSLT-2016 as follow: [data](https://drive.google.com/drive/folders/1gogdZW1AUolzVuYBm86r0cIiKo8m32aL?usp=sharing)

+ For training by ARA (3,4,2,2) in ResNet-152 on ImageNet
```bash
python train.py --dataset imagenet --model resnet152 --learning-rule ara --ARA-stride 3 4 2 2 --device 0 1 2 --imagenet_data_path PATH
```

+ For training by ARA (3,4,2,2) with GCP
```bash
python train.py --dataset imagenet --model resnet152 --learning-rule ara --ARA-stride 3 4 2 2 --device 0 1 2 --imagenet_data_path PATH --gcp
```

+ For training by ARA (3,4,2,2) with ActNN
```bash
python train.py --dataset imagenet --model resnet152 --learning-rule ara --ARA-stride 3 4 2 2 --device 0 1 2 --imagenet_data_path PATH --actnn
```

# Acknowledgments

 In this repository, code of [pytorch-resnet](https://github.com/kuangliu/pytorch-cifar) are modified to apply our AAL. Thanks the authors for open-source code.
