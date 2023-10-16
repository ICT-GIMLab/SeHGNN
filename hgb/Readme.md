# SeHGNN on the four medium-scale datasets

## Training

To reproduce the results of SeHGNN on four medium-scale datasets, please run following commands.

For **DBLP**:

```bash
python main.py --epoch 200 --dataset DBLP --n-fp-layers 2 --n-task-layers 3 --num-hops 2 --num-label-hops 4 \
	--label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5
```

For **ACM**:

```bash
python main.py --epoch 200 --dataset ACM --n-fp-layers 2 --n-task-layers 1 --num-hops 4 --num-label-hops 4 \
	--label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --amp --seeds 1 2 3 4 5
```

For **IMDB**:

```bash
python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
	--label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --amp --seeds 1 2 3 4 5
```

For **Freebase**:

```bash
python main.py --epoch 200 --dataset Freebase --n-fp-layers 2 --n-task-layers 4 --num-hops 2 --num-label-hops 3 \
	--label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 \
	--lr 3e-5 --weight-decay 3e-5 --batch-size 256 --amp --patience 30 --seeds 1 2 3 4 5
```

## Performance

![image-sehgnn-middle](./image-sehgnn_hgb.png)
