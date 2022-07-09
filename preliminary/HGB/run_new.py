import os
import sys
sys.path.append('../../')
import time
import argparse
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
import numpy as np
import random

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from GNN import myGAT, myGAT2
import dgl
import uuid
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)

    print(f'Show dataset {args.dataset}')
    print(dl.nodes['count'])
    print(dl.links)

    if args.dataset == 'Freebase':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    labels = torch.LongTensor(labels).to(device)

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    if not os.path.exists(f'pre_data_{args.dataset}.pt'):
        edge2type = {}
        for k in dl.links['data']:
            for u,v in zip(*dl.links['data'][k].nonzero()):
                edge2type[(u,v)] = k
        for i in range(dl.nodes['total']):
            if (i,i) not in edge2type:
                edge2type[(i,i)] = len(dl.links['count'])
        for k in dl.links['data']:
            for u,v in zip(*dl.links['data'][k].nonzero()):
                if (v,u) not in edge2type:
                    edge2type[(v,u)] = k+1+len(dl.links['count'])

        e_feat = []
        for u, v in zip(*g.edges()):
            u = u.cpu().item()
            v = v.cpu().item()
            e_feat.append(edge2type[(u,v)])
        e_feat = torch.tensor(e_feat, dtype=torch.long)

        torch.save(e_feat, f'pre_data_{args.dataset}.pt')
    else:
        e_feat = torch.load(f'pre_data_{args.dataset}.pt')
    e_feat = e_feat.to(device)

    num_relations = e_feat.max().item() + 1

    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT2(g, args.edge_feats, len(dl.links['count'])*2+1,
            in_dims, args.hidden_dim, num_classes, args.num_layers, heads,
            F.elu, args.dropout, args.dropout, args.slope, True, 0.05, num_relations=num_relations)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        net.train()
        save_path = 'checkpoint'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/checkpoint_{args.dataset}_{args.num_layers}_{uuid.uuid4().hex}.pkl'
        print('Model will save to', save_path)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=save_path)
        train_times = []
        for epoch in range(args.epoch):
            torch.cuda.synchronize()
            t_start = time.time()
            net.train()

            if args.average_attention_values:
                logits = net(features_list, e_feat, average_weight_layers=[0,1,2])
            else:
                logits = net(features_list, e_feat)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t_end = time.time()
            if epoch > 5:
                train_times.append(t_end - t_start)

            train_acc = [
                f1_score(labels[train_idx].cpu().squeeze(), logits[train_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[train_idx].cpu().squeeze(), logits[train_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('train_acc', train_acc)

            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()
            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_feat)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            t_end = time.time()
            val_acc = [
                f1_score(labels[val_idx].cpu().squeeze(), logits[val_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[val_idx].cpu().squeeze(), logits[val_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('val_acc', val_acc)
            test_acc = [
                f1_score(labels[test_idx].cpu().squeeze(), logits[test_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[test_idx].cpu().squeeze(), logits[test_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('test_acc', test_acc)
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))

            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print('average train times', sum(train_times) / len(train_times))

        net.load_state_dict(torch.load(save_path), strict=True)
        net.eval()

        if not args.average_attention_values:
            print('\n\nThe result of original HGB is:')
            with torch.no_grad():
                logits = net(features_list, e_feat, average_weight_layers=[])
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            val_acc = [
                f1_score(labels[val_idx].cpu().squeeze(), logits[val_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[val_idx].cpu().squeeze(), logits[val_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('\tval_acc', val_acc)
            test_acc = [
                f1_score(labels[test_idx].cpu().squeeze(), logits[test_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[test_idx].cpu().squeeze(), logits[test_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('\ttest_acc', test_acc)

        if True:
            if not args.average_attention_values:
                print('\nThe result of HGB* is:')
            else:
                print('\nThe result of HGB† is:')

            with torch.no_grad():
                logits = net(features_list, e_feat, average_weight_layers=[0,1,2])
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            val_acc = [
                f1_score(labels[val_idx].cpu().squeeze(), logits[val_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[val_idx].cpu().squeeze(), logits[val_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('val_acc', val_acc)
            test_acc = [
                f1_score(labels[test_idx].cpu().squeeze(), logits[test_idx].cpu().argmax(dim=-1), average='micro'),
                f1_score(labels[test_idx].cpu().squeeze(), logits[test_idx].cpu().argmax(dim=-1), average='macro')
            ]
            print('test_acc', test_acc)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('-s', '--seed',  nargs='+', type=int, default=[1], help='Random seeds')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)

    ap.add_argument('--root', type=str, default='../../data')
    ap.add_argument('--average-attention-values', action='store_true', default=False)

    args = ap.parse_args()
    # run_model_DBLP(args)
    print(args)
    for seed in args.seed:
        if seed > 0:
            set_random_seed(seed)
        run_model_DBLP(args)