# Experiments on ViT_L

See help (--h flag) for available options before executing the code.

+ For training by ASA4 in ViT_L on Cifar-100
```bash
python  train.py --name asa4_exp --dataset cifar10 --model_type ViT-L_32 --pretrained_dir PATH --device 0 1 2 --learning_rate 1e-1 --learning-rule asa4

```
+ For training by ASA4 + Mesa
```bash
python  train.py --name asa4_exp --dataset cifar10 --model_type ViT-L_32 --pretrained_dir PATH --device 0 1 2 --learning_rate 1e-1 --learning-rule asa4 --mesa --mesa_policy PATH
```


# Acknowledgments

 In this repository, code of [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are modified to apply our AAL. Thanks the authors for open-source code.
