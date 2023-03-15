# Experiments on Transformer

We prepaired Tiny-ImageNet and IWSLT-2016 as follow: [data](https://drive.google.com/drive/folders/1gogdZW1AUolzVuYBm86r0cIiKo8m32aL?usp=sharing)

See help (--h flag) for available options before executing the code.

+ For training by ASA4 in Transformer on IWSLT
```bash
python train.py --batch_size 4096 --dataset_name IWSLT --language_direction G2E --learning_rule asa4 --dataset_path DATA_PATH --device 0
```

+ For training by ASA4 in Transformer with Mesa
```bash
python train.py --batch_size 4096 --dataset_name IWSLT --language_direction G2E --learning_rule asa4 --dataset_path DATA_PATH --mesa True --mesa_policy POLICY_PATH --device 0
```

# Acknowledgments

 In this repository, code of [pytorch-original-transformer](https://github.com/gordicaleksa/pytorch-original-transformer) are modified to apply our AAL. Thanks the authors for open-source code.
