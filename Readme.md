# Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN)

## Requirements

#### 1. Some mainstream neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* [dgl](https://www.dgl.ai/pages/start.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the newest versions of these libraries.

#### 2. Other dependencies

Install other requirements:

```setup
pip install -r requirements.txt
```

Compile and install `sparse-tools`. Under the folder `./sparse_tools/`, run

```bash
python setup.py develop
```

`sparse-tools` is implemented for acceleration of label propagation for large dataset such as ogbn-mag.

## Data preparation

For the preliminary experiments and experiments on four middle-scale datasets, please download datasets `DBLP.zip`, `ACM.zip`, `IMDB.zip`, `Freebase.zip` from [the source of HGB benchmark](https://cloud.tsinghua.edu.cn/d/fc10cb35d19047a88cb1/?p=NC), and extract content from these compresesed files under the folder `'./data/'`.

For the experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from [OGB Challenge](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag).

## Experiments

For the preliminary experiments on HAN and HGB in Section 4 of the paper, please refer to folders `./preliminary/HAN/` and `./preliminary/HGB/`, respectively.

For the experiments on four middle-scale datasets in Section 6 of the paper, please refer to the folder `./middle/`.

For the experiments on the large dataset ogbn-mag in Section 6 of the paper, please refer to the folder `./large/`.

## Acknowledgement

This repository benefits a lot from [HGB](https://github.com/THUDM/HGB) and [GAMLP](https://github.com/PKU-DAIR/GAMLP).
