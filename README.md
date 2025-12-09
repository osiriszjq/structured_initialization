# Structured Initialization for Vision Transformers
### [Project Page](https://osiriszjq.github.io/structured_initialization) | [Paper](https://arxiv.org/abs/2505.19985)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[Jianqiao Zheng](https://github.com/osiriszjq/),
[Xueqian Li](https://lilac-lee.github.io/),
[Hemanth Saratchandran](https://scholar.google.com/citations?user=2pBBnegAAAAJ&hl=en),
[Simon Lucey](https://www.adelaide.edu.au/directory/simon.lucey)<br>
The University of Adelaide

This is the official implementation of the paper "Structured Initialization for Vision Transformers", which has been accepted to NeurIPS 2025.

## Code overview

Our code is based on the `timm` framework, which can be download from [here](http://github.com/rwightman/pytorch-image-models). We test our code on v1.0.22, but it should also work on other versions.

The implementation of our method can be found in `vision_transformer.py`, which is in [this commit](https://github.com/osiriszjq/structured_initialization/commit/7cd02f1a12d1d40a5cc4a589dff2bd46c62f5a2c).

🔎 For previous version without using `timm` framework, please check out [this repository](https://github.com/osiriszjq/impulse_init), which also contains a colab of our motivation.

## Usage

Run the following command under `pytorch-image-models-1.0.22` folder.

### ImageNet-1K

```
torchrun --nproc_per_node=16 train.py [/path/to/your/ImageNet-1k]
    --train-split train 
    --val-split val
    --input-size 3 224 224
    --mean 0.485 0.456 0.406
    --std 0.229 0.224 0.225
    --num-classes 1000
    --seed 42
    --model vit_base_patch16_224
    --model-kwargs img_size=224 weight_init=skip post_weight_init=[initialization/method]
    -j 10 
    -b 64
    --lr-base-size 512
    --lr-base 5e-4
    --lr-base-scale linear
    --warmup-lr 1e-6
    --min-lr 1e-5
    --weight-decay 0.05
    --opt adamw
    --opt-eps 1e-8
    --epochs 300
    --sched cosine
    --warmup-epochs 5
    --cooldown-epochs 10
    --amp
    --aa rand-m9-mstd0.5-inc1
    --cutmix 1.0
    --mixup 0.8
    --reprob 0.25
    --smoothing 0.1
    --drop 0.0
    --color-jitter 0.3
    --drop-path 0.1
    --crop-pct 0.875
    --aug-repeats 3.0
    --pin-mem
```

You can replace `[initialization/method]` with `default`, `mimetic` or `impulse`. Remember to set `[/path/to/your/ImageNet-1k]`.

### Small Datasets

```
torchrun --nproc_per_node=8 train.py [/path/to/your/data/folder]
    --dataset torch/[dataset/name]
    --dataset-download
    --input-size 3 224 224
    --mean 0.485 0.456 0.406
    --std 0.229 0.224 0.225
    --seed 0
    --num-classes [num/classes]
    --model vit_tiny_patch16_224
    --model-kwargs img_size=224 weight_init=skip post_weight_init=[initialization/method]
    --model-ema-decay 0.9999 
    -j 10
    -b 64
    --lr 2e-3
    --layer-decay 1.0
    --warmup-lr 1e-6
    --min-lr 1e-6
    --weight-decay 0.05
    --opt adamw
    --opt-eps 1e-8
    --epochs 300
    --sched cosine
    --warmup-epochs 50
    --amp
    --aa rand-m9-mstd0.5-inc1
    --cutmix 1.0
    --mixup 0.8
    --reprob 0.25
    --smoothing 0.1
    --drop 0.0
    --color-jitter 0.4
    --drop-path 0.1
    --crop-pct 0.875
    --pin-mem
```

You need to set `[/path/to/your/data/folder]` and choose the dataset you want to replace `[dataset/name]`, like `torch/CIFAR10`. Remember to set `[num/classes]`. You can replace `[initialization/method]` with `default`, `mimetic` or `impulse`.


## Citation
```
@article{zheng2025structured,
  title={Structured Initialization for Vision Transformers},
  author={Zheng, Jianqiao and Li, Xueqian and Saratchandran, Hemanth and Lucey, Simon},
  journal={arXiv preprint arXiv:2505.19985},
  year={2025}
}
```
