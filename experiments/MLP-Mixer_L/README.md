# Experiments on MLP-Mixer_L

See help (--h flag) for available options before executing the code.

+ For training by ASA4 in ViT_L on CIFAR-100
```bash
python3 train.py --name asa4_exp --model_type Mixer-L_16 --pretrained_dir PATH --device 0 1 2 3 4 --learning_rate 1e-1 --train_batch_size 256
```

# Acknowledgments

 In this repository, code of [MLP-Mixer-Pytorch](https://github.com/jeonsworld/MLP-Mixer-Pytorch) are modified to apply our AAL. Thanks the authors for open-source code.
