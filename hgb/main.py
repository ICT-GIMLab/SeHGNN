import os
import gc
import re
import time
import uuid
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

from model import *
from utils import *
from sparse_tools import SparseAdjList


def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)

    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full \
        = load_dataset(args)

    if not args.neighbor_attention:
        for k in adjs.keys():
            adjs[k].storage._value = None
            adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]

    # =======
    # rearange node idx (for feats & labels)
    # =======
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = train_node_nums + valid_node_nums + test_node_nums
    num_nodes = dl.nodes['count'][0]

    if total_num_nodes < num_nodes:
        flag = np.ones(num_nodes, dtype=bool)
        flag[train_nid] = 0
        flag[val_nid] = 0
        flag[test_nid] = 0
        extra_nid = np.where(flag)[0]
        print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
    else:
        extra_nid = np.array([])

    init2sort = torch.LongTensor(np.concatenate([train_nid, val_nid, test_nid, extra_nid]))
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]

    # =======
    # neighbor aggregation
    # =======
    if args.dataset == 'DBLP':
        tgt_type = 'A'
        node_types = ['A', 'P', 'T', 'V']
        extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        node_types = ['P', 'A', 'C']
        extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        node_types = ['M', 'A', 'D', 'K']
        extra_metapath = []
    elif args.dataset == 'Freebase':
        tgt_type = '0'
        node_types = [str(i) for i in range(8)]
        extra_metapath = []
    else:
        assert 0
    extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

    print(f'Current num hops = {args.num_hops}')

    if args.dataset == 'Freebase':
        prop_device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    else:
        prop_device = 'cpu'
    store_device = 'cpu'

    if args.dataset == 'Freebase':
        if not os.path.exists('./Freebase_adjs'):
            os.makedirs('./Freebase_adjs')
        num_tgt_nodes = dl.nodes['count'][0]

    # compute k-hop feature
    prop_tic = datetime.datetime.now()
    if args.dataset != 'Freebase':
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1

        if args.neighbor_attention:
            meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device='cpu')
            assert tgt_type not in meta_adjs
            raw_feats = {k: g.nodes[k].data[k].clone() for k in g.ndata.keys()}
            print(f'For tgt {tgt_type}, Involved raw_feat keys {raw_feats.keys()}, feats keys {meta_adjs.keys()}')
        elif args.two_layer:
            assert node_types[0] == tgt_type
            meta_adjs = hg_propagate_sparse_pyg(adjs, node_types, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device='cpu')
            for k in meta_adjs.keys(): assert len(k) > 1, k 
            raw_feats = {k: g.nodes[k].data[k].clone() for k in g.ndata.keys()}
            print(f'For tgt {tgt_type}, Involved raw_feat keys {raw_feats.keys()}, feats keys {meta_adjs.keys()}')
        else:
            g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, extra_metapath, echo=True)
            feats = {}
            keys = list(g.nodes[tgt_type].data.keys())
            print(f'For tgt {tgt_type}, feature keys {keys}')
            for k in keys:
                feats[k] = g.nodes[tgt_type].data.pop(k)
    else:
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1

        if args.two_layer:
            meta_adjs = hg_propagate_sparse_pyg(adjs, node_types, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device='cpu')
            for k in meta_adjs.keys(): assert len(k) > 1, k
        elif args.num_hops == 1:
            meta_adjs = {k: v.clone() for k, v in adjs.items() if k[0] == tgt_type}
        else:
            save_name = f'./Freebase_adjs/feat_seed{args.seed}_hop{args.num_hops}'
            if args.seed > 0 and os.path.exists(f'{save_name}_00_int64.npy'):
                # meta_adjs = torch.load(save_name)
                meta_adjs = {}
                for srcname in tqdm(dl.nodes['count'].keys()):
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    for k in tmp.keys:
                        assert k not in meta_adjs
                    meta_adjs.update(tmp.load_adjs(expand=True))
                    del tmp
            else:
                meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device=prop_device)

                meta_adj_list = []
                for srcname in dl.nodes['count'].keys():
                    keys = [k for k in meta_adjs.keys() if k[-1] == str(srcname)]
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', keys, meta_adjs, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    meta_adj_list.append(tmp)

                for srcname in dl.nodes['count'].keys():
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    tmp_adjs = tmp.load_adjs(expand=True)
                    print(srcname, tmp.keys)
                    for k in tmp.keys:
                        assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                        assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                        assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                    del tmp_adjs, tmp
                    gc.collect()

        feats = {k: v.clone() for k, v in meta_adjs.items() if len(k) <= args.num_hops + 1 or k in extra_metapath}
        if args.neighbor_attention:
            for k in feats.keys():
                feats[k].storage._value = None

        assert '0' not in feats
        if not args.neighbor_attention and not args.two_layer:
            feats['0'] = SparseTensor.eye(dl.nodes['count'][0])
        print(f'For tgt {tgt_type}, Involved keys {feats.keys()}')

    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        if args.neighbor_attention or args.two_layer:
            data_size = {k: g.ndata[k][k].size(-1) for k in g.ndata.keys()}
            raw_feats[tgt_type] = raw_feats[tgt_type][init2sort]

            feats = {}
            for k, v in tqdm(meta_adjs.items()):
                assert len(k) > 1
                if k[0] == tgt_type and k[-1] == tgt_type:
                    feats[k] = v[init2sort, init2sort]
                elif k[0] == tgt_type:
                    feats[k] = v[init2sort]
                else:
                    assert args.two_layer
                    if k[-1] == tgt_type:
                        feats[k] = v[:, init2sort]
                    else:
                        feats[k] = v
        else:
            data_size = {k: v.size(-1) for k, v in feats.items()}
            feats = {k: v[init2sort] for k, v in feats.items()}
    elif args.dataset == 'Freebase':
        data_size = dict(dl.nodes['count'])
        if args.neighbor_attention or args.two_layer:
            raw_feats = {}
            for k, count in data_size.items():
                raw_feats[k] = SparseTensor(row=torch.arange(count), col=torch.arange(count))

        for k, v in tqdm(feats.items()):
            if len(k) == 1:
                assert not args.neighbor_attention and not args.two_layer
                continue

            if k[0] == '0' and k[-1] == '0':
                # feats[k] = v[init2sort[:total_num_nodes], init2sort]
                # feats[k] = v[init2sort, init2sort]
                feats[k], _ = v.sample_adj(init2sort, -1, False) # faster, 50% time acceleration
            elif k[0] == '0':
                # feats[k] = v[init2sort[:total_num_nodes]]
                feats[k] = v[init2sort]
            else:
                assert args.two_layer, k
                if k[-1] == tgt_type:
                    feats[k] = v[:, init2sort]
                else:
                    feats[k] = v
    else:
        assert 0
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # =======
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    if args.dataset != 'IMDB':
        labels_cuda = labels.long().to(device)
    else:
        labels = labels.float()
        labels_cuda = labels.to(device)

    for stage in [0]:
        epochs = args.stage

        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        if args.label_feats:
            if args.dataset != 'IMDB':
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = F.one_hot(init_labels[train_nid], num_classes).float()
            else:
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = init_labels[train_nid].float()

            if args.dataset == 'DBLP':
                extra_metapath = []
            elif args.dataset == 'IMDB':
                extra_metapath = []
            elif args.dataset == 'ACM':
                extra_metapath = []
            elif args.dataset == 'Freebase':
                extra_metapath = []
            else:
                assert 0

            extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]
            if len(extra_metapath):
                max_length = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
            else:
                max_length = args.num_label_hops + 1

            print(f'Current label-prop num hops = {args.num_label_hops}')
            # compute k-hop feature
            prop_tic = datetime.datetime.now()
            if args.dataset == 'Freebase' and args.num_label_hops <= args.num_hops and len(extra_metapath) == 0:
                meta_adjs = {k: v for k, v in meta_adjs.items() if k[-1] == '0' and len(k) < max_length}
            else:
                if args.dataset == 'Freebase':
                    save_name = f'./Freebase_adjs/label_seed{args.seed}_hop{args.num_label_hops}'
                    if args.seed > 0 and os.path.exists(f'{save_name}_int64.npy'):
                        meta_adj_list = SparseAdjList(save_name, None, None, num_tgt_nodes, num_tgt_nodes, with_values=True)
                        meta_adjs = meta_adj_list.load_adjs(expand=True)
                    else:
                        meta_adjs = hg_propagate_sparse_pyg(
                            adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)
                        meta_adj_list = SparseAdjList(save_name, meta_adjs.keys(), meta_adjs, num_tgt_nodes, num_tgt_nodes, with_values=True)

                        tmp = SparseAdjList(save_name, None, None, num_tgt_nodes, num_tgt_nodes, with_values=True)
                        tmp_adjs = tmp.load_adjs(expand=True)
                        for k in tmp.keys:
                            assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                            assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                            assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                        del tmp_adjs, tmp
                        gc.collect()
                else:
                    meta_adjs = hg_propagate_sparse_pyg(
                        adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)

            if args.dataset == 'Freebase':
                if 0:
                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()

                    condition = lambda ra,rb,rc,k: rb > 0.2
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False)

                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(label_feats.keys())) - set(left_keys))
                    for k in remove_keys:
                        label_feats.pop(k)
                else:
                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(meta_adjs.keys())) - set(left_keys))
                    for k in remove_keys:
                        meta_adjs.pop(k)

                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                for k, v in tqdm(meta_adjs.items()):
                    if args.dataset != 'Freebase':
                        label_feats[k] = remove_diag(v) @ label_onehot
                    else:
                        label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)
                gc.collect()

                if args.dataset == 'IMDB':
                    condition = lambda ra,rb,rc,k: True
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False, loss_type='bce')
                else:
                    condition = lambda ra,rb,rc,k: True
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=True)
            print('Involved label keys', label_feats.keys())

            label_feats = {k: v[init2sort] for k,v in label_feats.items()}
            prop_toc = datetime.datetime.now()
            print(f'Time used for label prop {prop_toc - prop_tic}')

        # =======
        # Train & eval loaders
        # =======
        train_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

        # =======
        # Mask & Smooth
        # =======
        with_mask = False
        # if args.dataset == 'Freebase':
        #     init_mask = {k: v.storage.rowcount() != 0 for k, v in feats.items()}
        #     with_mask = True
        # else:
        #     print(f'TODO: `with_mask` has not be implemented for {args.dataset}')

        # if with_mask:
        #     train_mask = {k: (v[:total_num_nodes] & (torch.randn(total_num_nodes) > 0)).float() for k, v in init_mask.items()}
        #     full_mask = {k: v.float() for k, v in init_mask.items()}
        # else:
        #     train_mask = full_mask = None

        # Freebase train/val/test/full_nodes: 1909/477/5568/40402
        # IMDB     train/val/test/full_nodes: 1097/274/3202/359
        eval_loader, full_loader = [], []
        batchsize = 2 * args.batch_size

        if args.two_layer:
            for batch_idx in range((total_num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize
                batch_end = min(total_num_nodes, (batch_idx+1) * batchsize)
                batch = torch.arange(batch_start, batch_end)

                layer2_feats = {k: x[batch_start:batch_end] for k, x in feats.items() if k[0] == tgt_type}
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}

                involved_keys = {}
                for k, v in layer2_feats.items():
                    src = k[-1]
                    if src not in involved_keys:
                        involved_keys[src] = []
                    involved_keys[src].append(torch.unique(v.storage.col()))
                involved_keys = {k: torch.unique(torch.cat(v)) for k, v in involved_keys.items()}

                for k, v in layer2_feats.items():
                    src = k[-1]
                    old_nnz = v.nnz()
                    layer2_feats[k] = v[:, involved_keys[src]]
                    assert layer2_feats[k].nnz() == old_nnz

                layer1_feats = {k: v[involved_keys[k[0]]] for k, v in feats.items() if k[0] in involved_keys}

                eval_loader.append((involved_keys, layer1_feats, batch, layer2_feats, batch_labels_feats))

            for batch_idx in range((num_nodes-total_num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize + total_num_nodes
                batch_end = min(num_nodes, (batch_idx+1) * batchsize + total_num_nodes)
                batch = torch.arange(batch_start, batch_end)

                layer2_feats = {k: x[batch_start:batch_end] for k, x in feats.items() if k[0] == tgt_type}
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}

                involved_keys = {}
                for k, v in layer2_feats.items():
                    src = k[-1]
                    if src not in involved_keys:
                        involved_keys[src] = []
                    involved_keys[src].append(torch.unique(v.storage.col()))
                involved_keys = {k: torch.unique(torch.cat(v)) for k, v in involved_keys.items()}

                for k, v in layer2_feats.items():
                    src = k[-1]
                    old_nnz = v.nnz()
                    layer2_feats[k] = v[:, involved_keys[src]]
                    assert layer2_feats[k].nnz() == old_nnz

                layer1_feats = {k: v[involved_keys[k[0]]] for k, v in feats.items() if k[0] in involved_keys}

                full_loader.append((involved_keys, layer1_feats, batch, layer2_feats, batch_labels_feats))
        else:
            for batch_idx in range((total_num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize
                batch_end = min(total_num_nodes, (batch_idx+1) * batchsize)
                batch = torch.arange(batch_start, batch_end)

                batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
                if with_mask:
                    batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                else:
                    batch_mask = None
                eval_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

            for batch_idx in range((num_nodes-total_num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize + total_num_nodes
                batch_end = min(num_nodes, (batch_idx+1) * batchsize + total_num_nodes)
                batch = torch.arange(batch_start, batch_end)

                batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
                if with_mask:
                    batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                else:
                    batch_mask = None
                full_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        # =======
        # Construct network
        # =======
        torch.cuda.empty_cache()
        gc.collect()
        if args.neighbor_attention:
            model = SeHGNN_NA(args.embed_size, args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
                args.dropout, args.input_drop, args.att_drop, args.label_drop,
                args.n_layers_1, args.n_layers_2, args.act, args.residual, bns=args.bns, data_size=data_size, num_heads=args.num_heads)
        elif args.two_layer:
            model = SeHGNN_2L(args.embed_size, args.hidden, num_classes,
                feats.keys(), [k for k in feats.keys() if k[0] == tgt_type], label_feats.keys(), node_types,
                args.dropout, args.input_drop, args.att_drop, args.label_drop,
                args.n_layers_1, args.n_layers_2, args.act, args.residual, bns=args.bns, data_size=data_size)
        else:
            model = SeHGNN(args.embed_size, args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
                args.dropout, args.input_drop, args.att_drop, args.label_drop,
                args.n_layers_1, args.n_layers_2, args.act, args.residual, bns=args.bns, data_size=data_size,
                remove_transformer=args.remove_transformer, independent_attn=args.independent_attn)
        model = model.to(device)
        if args.seed == args.seeds[0]:
            print(model)
            print("# Params:", get_n_params(model))

        if args.dataset == 'IMDB':
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        best_epoch = -1
        best_val_loss = 1000000
        best_test_loss = 0
        best_val = (0,0)
        best_test = (0,0)
        val_loss_list, test_loss_list = [], []
        val_acc_list, test_acc_list = [], []
        actual_loss_list, actual_acc_list = [], []
        store_list = []
        best_pred = None
        count = 0

        train_times = []

        if args.neighbor_attention or args.two_layer:
            model.feats = {k: v.to(device) for k, v in raw_feats.items()}

        for epoch in tqdm(range(args.stage)):
            gc.collect()
            torch.cuda.synchronize()
            start = time.time()
            if args.two_layer:
                loss, acc = train_2l(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, tgt_type, scalar=scalar)
            else:
                loss, acc = train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, scalar=scalar)
            torch.cuda.synchronize()
            end = time.time()

            log = "Epoch {}, training Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}, {:.4f}\n".format(epoch, end - start,loss, acc[0]*100, acc[1]*100)
            torch.cuda.empty_cache()
            train_times.append(end-start)

            start = time.time()
            with torch.no_grad():
                model.eval()
                raw_preds = []
                if args.two_layer:
                    for batch1, layer1_feats, batch2, layer2_feats, batch_labels_feats in eval_loader:
                        batch1 = {k: v.to(device) for k,v in batch1.items()}
                        layer1_feats = {k: v.to(device) for k,v in layer1_feats.items()}
                        batch2 = batch2.to(device)
                        layer2_feats = {k: v.to(device) for k,v in layer2_feats.items()}
                        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                        raw_preds.append(model(layer1_feats, batch1, layer2_feats, batch2, batch_labels_feats).cpu())
                else:
                    for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
                        batch = batch.to(device)
                        batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                        if with_mask:
                            batch_mask = {k: x.to(device) for k, x in batch_mask.items()}
                        else:
                            batch_mask = None
                        raw_preds.append(model(batch, batch_feats, batch_labels_feats, batch_mask).cpu())

                raw_preds = torch.cat(raw_preds, dim=0)
                loss_train = loss_fcn(raw_preds[:trainval_point], labels[:trainval_point]).item()
                loss_val = loss_fcn(raw_preds[trainval_point:valtest_point], labels[trainval_point:valtest_point]).item()
                loss_test = loss_fcn(raw_preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes]).item()

            if args.dataset != 'IMDB':
                preds = raw_preds.argmax(dim=-1)
            else:
                preds = (raw_preds > 0.).int()

            train_acc = evaluator(preds[:trainval_point], labels[:trainval_point])
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[trainval_point:valtest_point])
            test_acc = evaluator(preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes])

            end = time.time()
            log += f'evaluation Time: {end-start}, Train loss: {loss_train}, Val loss: {loss_val}, Test loss: {loss_test}\n'
            log += 'Train acc: ({:.4f}, {:.4f}), Val acc: ({:.4f}, {:.4f}), Test acc: ({:.4f}, {:.4f}) ({})\n'.format(
                train_acc[0]*100, train_acc[1]*100, val_acc[0]*100, val_acc[1]*100, test_acc[0]*100, test_acc[1]*100, total_num_nodes-valtest_point)

            if (args.dataset != 'Freebase' and loss_val <= best_val_loss) or (args.dataset == 'Freebase' and sum(val_acc) >= sum(best_val)):
                best_epoch = epoch
                best_val_loss = loss_val
                best_test_loss = loss_test
                best_val = val_acc
                best_test = test_acc

                best_pred = raw_preds
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')

                if epoch - best_epoch > args.patience: break

            if epoch > 0 and epoch % 10 == 0: 
                log = log + f'\tCurrent best at epoch {best_epoch} with Val loss {best_val_loss:.4f} ({best_val[0]*100:.4f}, {best_val[1]*100:.4f})' \
                    + f', Test loss {best_test_loss:.4f} ({best_test[0]*100:.4f}, {best_test[1]*100:.4f})'
            print(log)

        print('average train times', sum(train_times) / len(train_times))

        print(f'Best Epoch {best_epoch} at {checkpt_file.split("/")[-1]}\n\tFinal Val loss {best_val_loss:.4f} ({best_val[0]*100:.4f}, {best_val[1]*100:.4f})'
            + f', Test loss {best_test_loss:.4f} ({best_test[0]*100:.4f}, {best_test[1]*100:.4f})')

        if len(full_loader):
            model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                raw_preds = []
                if args.two_layer:
                    for batch1, layer1_feats, batch2, layer2_feats, batch_labels_feats in full_loader:
                        batch1 = {k: v.to(device) for k,v in batch1.items()}
                        layer1_feats = {k: v.to(device) for k,v in layer1_feats.items()}
                        batch2 = batch2.to(device)
                        layer2_feats = {k: v.to(device) for k,v in layer2_feats.items()}
                        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                        raw_preds.append(model(layer1_feats, batch1, layer2_feats, batch2, batch_labels_feats).cpu())
                else:
                    for batch, batch_feats, batch_labels_feats, batch_mask in full_loader:
                        batch = batch.to(device)
                        batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                        if with_mask:
                            batch_mask = {k: x.to(device) for k, x in batch_mask.items()}
                        else:
                            batch_mask = None
                        raw_preds.append(model(batch, batch_feats, batch_labels_feats, batch_mask).cpu())
                raw_preds = torch.cat(raw_preds, dim=0)
            best_pred = torch.cat((best_pred, raw_preds), dim=0)

        torch.save(best_pred, f'{checkpt_file}.pt')

        if args.dataset != 'IMDB':
            predict_prob = best_pred.softmax(dim=1)
        else:
            predict_prob = torch.sigmoid(best_pred)

        test_logits = predict_prob[sort2init][test_nid_full]
        if args.dataset != 'IMDB':
            pred = test_logits.cpu().numpy().argmax(axis=1)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred, file_path=f"{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt")
        else:
            pred = (test_logits.cpu().numpy()>0.5).astype(int)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred, file_path=f"{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt", mode='multi')

    if args.dataset != 'IMDB':
        preds = predict_prob.argmax(dim=1, keepdim=True)
    else:
        preds = (predict_prob > 0.5).int()
    train_acc = evaluator(labels[:trainval_point], preds[:trainval_point])
    val_acc = evaluator(labels[trainval_point:valtest_point], preds[trainval_point:valtest_point])
    test_acc = evaluator(labels[valtest_point:total_num_nodes], preds[valtest_point:total_num_nodes])

    print(f'train_acc ({train_acc[0]*100:.2f}, {train_acc[1]*100:.2f}) ' \
        + f'val_acc ({val_acc[0]*100:.2f}, {val_acc[1]*100:.2f}) ' \
        + f'test_acc ({test_acc[0]*100:.2f}, {test_acc[1]*100:.2f})')
    print(checkpt_file.split('/')[-1])


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default="../data/")
    parser.add_argument("--stage", type=int, default=200, help="The epoch setting for each stage.")
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-1", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--n-layers-2", type=int, default=3,
                        help="number of layers of the downstream task")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,
                        help="label feature dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--label-bns", action='store_true', default=False,
                        help="whether to process the input label features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--drop-metapath", type=float, default=0,
                        help="whether to process the input features")
    ## for ablation
    parser.add_argument("-na", "--neighbor-attention", action='store_true', default=False)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--two-layer", action='store_true', default=False)
    parser.add_argument("--remove-transformer", action='store_true', default=False)
    parser.add_argument("--independent-attn", action='store_true', default=False)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    # args.bns = args.bns and args.dataset == 'Freebase' # remove bn for full-batch learning
    if args.dataset == 'ACM':
        args.ACM_keep_F = False
    assert args.neighbor_attention + args.two_layer + args.remove_transformer <= 1

    args.seed = args.seeds[0]
    print(args)

    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        main(args)