import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import sys

def load_data(prefix='DBLP', raw_dir='', load_full_test=False):
    if raw_dir == '':
        raw_dir = '../../data'
    sys.path.append(raw_dir)
    from data_loader import data_loader
    dl = data_loader(f'{raw_dir}/{prefix}')
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    if not load_full_test:
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
    else:
        test_idx = np.nonzero(dl.labels_test['mask'] | dl.labels_test_full['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels[test_idx] = dl.labels_test['data'][test_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl
