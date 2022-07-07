# SeHGNN on Ogbn-mag

For environment setup, please refer to the [main page of this repository](https://github.com/ICT-GIMLab/SeHGNN/#requirements).

## Training without extra embeddings

```setup
python main.py --stages 300 300 300 300 --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp
```

For the first time this command is executed, the dataset ogbn-mag will be automatically donwloaded in the folder `../data/`.

The codes generate random embeddings for node types author/topic/institution.

## Training with extra embeddings from ComplEx

For better model effect, we utilize [ComplEx](https://proceedings.mlr.press/v48/trouillon16.html) to generate embeddings for those three node types with no raw features.

Following [NARS](https://github.com/facebookresearch/NARS), we use `dglke` to generate embeddings. However, dglke only supports dgl<=0.4.3. We modify some files so that it fits the latest versions of dgl.

**1.Generate extra embeddings from ComplEx**

Please make sure that the ogbn-mag dataset has been downloaded in the folder `../data/`.

Then under the folder `../data/complex_nars`, run

```setup
python convert_to_triplets.py --dataset mag
bash train_graph_emb.sh mag
```

Check the running log to find where the generated ComplEx features is saved. For example, if the save folder is `ckpts/ComplEx_mag_0`, run

```setup
python split_node_emb.py --dataset mag --emb-file ckpts/ComplEx_mag_0/mag_ComplEx_entity.npy
```

**2.Training our sehgnn model**

Under this folder, run

```setup
python main.py --stages 300 300 300 300 --extra-embedding complex --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp
```

## Performance

![image-sehgnn_mag](./image-sehgnn_mag.png)