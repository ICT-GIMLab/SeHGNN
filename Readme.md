# Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN)

The camera-ready paper for AAAI 23 can be found at: [http://arxiv.org/abs/2207.02547](http://arxiv.org/abs/2207.02547)

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

* [dgl](https://www.dgl.ai/pages/start.html)

If you want to generate ComplEx embeddings for ogbn-mag, we recommend to install `dgl<1.0`.

#### 2. Other dependencies

Install other requirements:

```setup
pip install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..
```

## Data preparation

* HGB datasets for node classification

These datasets include four medium-scale datasets. Please download them (`DBLP.zip`, `ACM.zip`, `IMDB.zip`, `Freebase.zip`) from [HGB repository](https://github.com/THUDM/HGB) and extract content under the folder `'./data/'`.

* Ogbn-mag

It is a large dataset from [OGB challenge](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag). Thus dataset will be automatically downloaded for the first time running.

---

If you encounter any issues, please feel free to reach out to me at yangxc96@gmail.com. The previous email address, yangxiaocheng@ict.ac.cn, is no longer in use.
