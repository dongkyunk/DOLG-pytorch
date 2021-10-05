
# Pytorch Implementation of Deep Local Feature (DeLF)

This is the unofficial PyTorch Implementation of "DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features"

reference: https://arxiv.org/pdf/2108.02927.pdf


## Prerequisites

+ PyTorch
+ PyTorch Lightning
+ timm
+ sklearn
+ pandas
+ jpeg4py
+ albumentations
+ python3
+ CUDA

## Data

You can get the GLDv2 dataset from [here](https://github.com/cvdfoundation/google-landmark).

If you just want the GLDv2-clean dataset, check this [kaggle competition dataset](https://www.kaggle.com/c/landmark-retrieval-2021).

Place your data like the structure below

```
data
├── train_clean.csv
└── train
    └── ###
        └── ###
            └── ###
                └── ###.jpg
```

## Citations

```bibtex
@misc{yang2021dolg,
      title={DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features}, 
      author={Min Yang and Dongliang He and Miao Fan and Baorong Shi and Xuetong Xue and Fu Li and Errui Ding and Jizhou Huang},
      year={2021},
      eprint={2108.02927},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```