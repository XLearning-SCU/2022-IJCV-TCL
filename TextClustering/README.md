PyTorch implementation for text clustering.

## Requirements

- sentence-transformers 1.0.3
- sentencepiece 0.1.95
- tokenizers 0.10.1
- transformers 4.4.2
- urllib3 1.26.4

## Datasets

### Biomedical and StackOverflow

We follow [SCCL](https://arxiv.org/abs/2103.12953) to obtain datasets.

## Training and Evaluation

### Pretrain and Boost

```train
# Pretrain
python train.py --batch_size 256 --epochs 500 --save_freq 100 --dataset Biomedical --model_path ./save/Biomedical/ --gpu 0

# Boost
python boost.py --batch_size 256 --epochs 510 --save_freq 1 --dataset Biomedical --model_path ./save/Biomedical/release-text/ --resume True --start_epoch 500 --gpu 0
```

### Evaluation

```
python evaluation.py --dataset Biomedical --eval_epoch 506 --model_path ./save/Biomedical/ --gpu 0
```

## License

[Apache License 2.0](
