# Experiments on BERT_L

Prepair data fir training BERT by [glue](https://github.com/nyu-mll/GLUE-baselines)

BERT_L(Uncased) can be download by [bert](https://github.com/google-research/bert)

See help (--h flag) for available options before executing the code.

+ For training by ASA4 in BERT_L on MNLI  (train)
```bash
export GLUE_DIR=GLUE_DIR
export BERT_PRETRAIN=BERT_PRETRAIN
export SAVE_DIR=SAVE_DIR

python classify.py --task mnli --mode train --train_cfg config/train_mnli.json --model_cfg config/bert_large.json --data_file $GLUE_DIR/MNLI/train.tsv --pretrain_file $BERT_PRETRAIN/bert_model.ckpt --vocab $BERT_PRETRAIN/vocab.txt --save_dir $SAVE_DIR --max_len 128 --learning_rule asa4
```
+ For training by ASA4 + Mesa  (train)
```bash
python classify.py --task mnli --mode train --train_cfg config/train_mnli.json --model_cfg config/bert_large.json --data_file $GLUE_DIR/MNLI/train.tsv --pretrain_file $BERT_PRETRAIN/bert_model.ckpt --vocab $BERT_PRETRAIN/vocab.txt --save_dir $SAVE_DIR --max_len 128 --learning_rule asa4 --mesa --mesa_policy PATH
```
+ For evaluating by ASA4 (eval)
```bash
python classify.py  --task mnli --mode eval --train_cfg config/train_mnli.json --model_cfg config/bert_large.json --data_file $GLUE_DIR/MNLI/dev_mismatched.tsv --model_file $SAVE_DIR/model_steps_36816.pt --vocab $BERT_PRETRAIN/vocab.txt --max_len 128 --learning_rule asa4
```

# Acknowledgments

 In this repository, code of [pytorchic-bert](https://github.com/dhlee347/pytorchic-bert) are modified to apply our AAL. Thanks the authors for open-source code.
