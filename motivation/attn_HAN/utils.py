import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import copy
import torch as th
import torch.nn.functional as F
import uuid

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

import sys

sys.path.append('../../data/')
from data_loader import data_loader


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args.log_dir,
        '{}_{}'.format(args.dataset, date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


def setup(args):
    args.update(default_configure)
    set_random_seed(args.seed)
    args.log_dir = setup_log_dir(args)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()


def load_acm(feat_type=0):
    assert feat_type == 0
    data_path = '../../data/ACM.mat'

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']
    p_vs_a = data['PvsA']
    p_vs_t = data['PvsT']
    p_vs_c = data['PvsC']

    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')

    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    meta_paths = [['pa', 'ap'], ['pf', 'fp']]
    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask, meta_paths


def load_acm_hgb(feat_type=0):
    dl = data_loader('../../data/ACM')
    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        features = th.FloatTensor(np.eye(paper_num))

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)

    num_classes = 3

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pp'], ['-pp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_freebase(feat_type=1):
    dl = data_loader('../../data/Freebase')
    link_type_dic = {0: '00', 1: '01', 2: '03', 3: '05', 4: '06',
                     5: '11',
                     6: '20', 7: '21', 8: '22', 9: '23', 10: '25',
                     11: '31', 12: '33', 13: '35',
                     14: '40', 15: '41', 16: '42', 17: '43', 18: '44', 19: '45', 20: '46', 21: '47',
                     22: '51', 23: '55',
                     24: '61', 25: '62', 26: '63', 27: '65', 28: '66', 29: '67',
                     30: '70', 31: '71', 32: '72', 33: '73', 34: '75', 35: '77',
                     36: '-00', 37: '10', 38: '30', 39: '50', 40: '60',
                     41: '-11',
                     42: '02', 43: '12', 44: '-22', 45: '32', 46: '52',
                     47: '13', 48: '-33', 49: '53',
                     50: '04', 51: '14', 52: '24', 53: '34', 54: '-44', 55: '54', 56: '64', 57: '74',
                     58: '15', 59: '-55',
                     60: '16', 61: '26', 62: '36', 63: '56', 64: '-66', 65: '76',
                     66: '07', 67: '17', 68: '27', 69: '37', 70: '57', 71: '-77',
                     }
    book_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        if link_type_dic[link_type + 36][0] != '-':
            data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = dl.links['data'][link_type].T.nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        indices = np.vstack((np.arange(book_num), np.arange(book_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(book_num))
        features = th.sparse.FloatTensor(indices, values, th.Size([book_num, book_num]))

    labels = dl.labels_test['data'][:book_num] + dl.labels_train['data'][:book_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)

    num_classes = 7

    train_valid_mask = dl.labels_train['mask'][:book_num]
    test_mask = dl.labels_test['mask'][:book_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['00', '00'], ['01', '10'], ['05', '52', '20'], ['04', '40'], ['04', '43', '30'], ['06', '61', '10'],
                  ['07', '70'], ]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_dblp(feat_type=0):
    prefix = '../../data/DBLP'
    dl = data_loader(prefix)
    link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        features = th.FloatTensor(np.eye(author_num))

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)

    num_classes = 4

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    np.random.shuffle(train_valid_indices)
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_imdb(feat_type=0):
    prefix = '../../data/IMDB'
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        features = th.FloatTensor(np.eye(movie_num))

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)

    num_classes = 5

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, dl


def load_data(dataset, feat_type=0):
    load_fun = None
    if dataset == 'ACM':
        load_fun = load_acm
    if dataset == 'ACM_HGB':
        load_fun = load_acm_hgb
    elif dataset == 'Freebase':
        feat_type = 1
        load_fun = load_freebase
    elif dataset == 'DBLP':
        load_fun = load_dblp
    elif dataset == 'IMDB':
        load_fun = load_imdb
    return load_fun(feat_type=feat_type)


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}_'.format(
            dt.date(), dt.hour, dt.minute, dt.second) + uuid.uuid4().hex + '.pth'
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss): # and (acc <= self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss): # and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
