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
    g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = load_dataset(args)

    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]

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

    print(f'Current num hops = {args.num_hops} for feature propagation')

    # If out of memory when using gpu, please set `prop_device = 'cpu'`
    if args.dataset == 'Freebase':
        prop_device = f'cuda:{args.gpu}' if not args.cpu else 'cpu'
    else:
        prop_device = 'cpu'
    store_device = 'cpu'

    if args.dataset == 'Freebase':
        if not os.path.exists('./Freebase_adjs'):
            os.makedirs('./Freebase_adjs')
        num_tgt_nodes = dl.nodes['count'][0]

    # compute k-hop features
    prop_tic = datetime.datetime.now()
    if args.dataset != 'Freebase':
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1

        g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, extra_metapath, echo=True)
        raw_feats = {}
        keys = list(g.nodes[tgt_type].data.keys())

        for k in keys:
            raw_feats[k] = g.nodes[tgt_type].data.pop(k)
    else:
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1

        if args.num_hops == 1:
            raw_meta_adjs = {k: v.clone() for k, v in adjs.items() if k[0] == tgt_type}
        else:
            save_name = f'./Freebase_adjs/feat_hop{args.num_hops}'
            if os.path.exists(f'{save_name}_00_int64.npy'):
                raw_meta_adjs = {}
                for srcname in dl.nodes['count'].keys():
                    print(f'Loading feature adjs from {save_name}_0{srcname}')
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    for k in tmp.keys:
                        assert k not in raw_meta_adjs
                    raw_meta_adjs.update(tmp.load_adjs(expand=True))
                    del tmp
            else:
                print('Generating feature adjs for Freebase ...\n(For each configutaion, this happens only once for the first execution, and results will be saved for later executions)')
                raw_meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device=prop_device)

                meta_adj_list = []
                for srcname in dl.nodes['count'].keys():
                    keys = [k for k in raw_meta_adjs.keys() if k[-1] == str(srcname)]
                    print(f'Saving feature adjs {keys} into {save_name}_0{srcname}')
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', keys, raw_meta_adjs, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    meta_adj_list.append(tmp)

                for srcname in dl.nodes['count'].keys():
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    tmp_adjs = tmp.load_adjs(expand=True)
                    print(srcname, tmp.keys)
                    for k in tmp.keys:
                        assert torch.all(raw_meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                        assert torch.all(raw_meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                        assert torch.all(raw_meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                    del tmp_adjs, tmp
                    gc.collect()

        raw_feats = {k: v.clone() for k, v in raw_meta_adjs.items() if len(k) <= args.num_hops + 1 or k in extra_metapath}

        assert '0' not in raw_feats
        raw_feats['0'] = SparseTensor.eye(dl.nodes['count'][0])

    print(f'For tgt type {tgt_type}, feature keys (num={len(raw_feats)}):', end='')
    if args.dataset == 'Freebase':
        print(' (in SparseTensor mode)')
        for k, v in raw_feats.items():
            print(k, v.sizes())
    else:
        print()
        for k, v in raw_feats.items():
            print(k, v.size())
    print()

    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        data_size = {k: v.size(-1) for k, v in raw_feats.items()}
    elif args.dataset == 'Freebase':
        data_size = dict(dl.nodes['count'])
    else:
        assert 0
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # =======
    if args.amp and not args.cpu:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    labels = init_labels.clone()

    device = f'cuda:{args.gpu}' if not args.cpu else 'cpu'
    if args.dataset != 'IMDB':
        labels_cuda = labels.long().to(device)
    else:
        labels = labels.float()
        labels_cuda = labels.to(device)

    for seed in args.seeds:
        print('Restart with seed =', seed)
        args.seed = seed
        set_random_seed(args.seed)

        checkpt_folder = f'./output/{args.dataset}/'
        if not os.path.exists(checkpt_folder):
            os.makedirs(checkpt_folder)
        checkpt_file = checkpt_folder + uuid.uuid4().hex
        print('checkpt_file', checkpt_file)

        val_ratio = 0.2
        train_nid = trainval_nid.copy()
        np.random.shuffle(train_nid)
        split = int(train_nid.shape[0]*val_ratio)
        val_nid = train_nid[:split]
        train_nid = train_nid[split:]
        train_nid = np.sort(train_nid)
        val_nid = np.sort(val_nid)

        train_node_nums = len(train_nid)
        valid_node_nums = len(val_nid)
        test_node_nums = len(test_nid)
        trainval_point = train_node_nums
        valtest_point = trainval_point + valid_node_nums
        print(f'#Train {train_node_nums}, #Val {valid_node_nums}, #Test {test_node_nums}')

        labeled_nid = np.concatenate((train_nid, val_nid, test_nid))
        labeled_num_nodes = len(labeled_nid)
        num_nodes = dl.nodes['count'][0]

        if labeled_num_nodes < num_nodes:
            flag = np.ones(num_nodes, dtype=bool)
            flag[train_nid] = 0
            flag[val_nid] = 0
            flag[test_nid] = 0
            extra_nid = np.where(flag)[0]
            print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
        else:
            extra_nid = np.array([])

        # clone raw_feats to avoid in-place modification for different seeds
        feats = {k: v.detach().clone() for k, v in raw_feats.items()}

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

            print(f'Current num hops = {args.num_label_hops} for label propagation')
            # compute k-hop feature
            prop_tic = datetime.datetime.now()
            if args.dataset == 'Freebase' and args.num_label_hops <= args.num_hops and len(extra_metapath) == 0:
                meta_adjs = {k: v for k, v in raw_meta_adjs.items() if k[-1] == '0' and len(k) < max_length}
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

            print(f'For label propagation, meta_adjs: (in SparseTensor mode)')
            for k, v in meta_adjs.items():
                print(k, v.sizes())
            print()

            if args.dataset == 'Freebase':
                if False:
                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    if not args.cpu: torch.cuda.empty_cache()
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
                    if not args.cpu: torch.cuda.empty_cache()
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

            # label_feats = {k: v[init2sort] for k,v in label_feats.items()}
            prop_toc = datetime.datetime.now()
            print(f'Time used for label prop {prop_toc - prop_tic}')

        # =======
        # Train & eval loaders
        # =======
        train_loader = torch.utils.data.DataLoader(
            train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)

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
        #     train_mask = {k: (v[:labeled_num_nodes] & (torch.randn(labeled_num_nodes) > 0)).float() for k, v in init_mask.items()}
        #     full_mask = {k: v.float() for k, v in init_mask.items()}
        # else:
        #     train_mask = full_mask = None

        # with_smooth = False

        # Freebase train/val/test/full_nodes: 1909/477/5568/40402
        # IMDB     train/val/test/full_nodes: 1097/274/3202/359
        eval_loader, full_loader = [], []
        eval_batch_size = 2 * args.batch_size

        for batch_idx in range((labeled_num_nodes-1) // eval_batch_size + 1):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(labeled_num_nodes, (batch_idx+1) * eval_batch_size)
            batch = torch.LongTensor(labeled_nid[batch_start:batch_end])

            batch_feats = {k: x[batch] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}
            if with_mask:
                batch_mask = {k: x[batch] for k, x in full_mask.items()}
            else:
                batch_mask = None
            eval_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        for batch_idx in range((len(extra_nid)-1) // eval_batch_size + 1):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(len(extra_nid), (batch_idx+1) * eval_batch_size)
            batch = torch.LongTensor(extra_nid[batch_start:batch_end])

            batch_feats = {k: x[batch] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}
            if with_mask:
                batch_mask = {k: x[batch] for k, x in full_mask.items()}
            else:
                batch_mask = None
            full_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        # =======
        # Construct network
        # =======
        if not args.cpu: torch.cuda.empty_cache()
        gc.collect()

        model = SeHGNN(args.dataset, args.embed_size, args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
                       args.dropout, args.input_drop, args.att_drop, args.n_fp_layers, args.n_task_layers, args.act,
                       args.residual, data_size=data_size)
        model = model.to(device)
        if args.seed == args.seeds[0]:
            print(model)
            print('# Params:', get_n_params(model))

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

        for epoch in tqdm(range(args.epoch)):
            gc.collect()
            if not args.cpu: torch.cuda.synchronize()
            start = time.time()
            loss, acc = train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, scalar=scalar)
            if not args.cpu: torch.cuda.synchronize()
            end = time.time()

            log = f'Epoch {epoch}, training Time(s): {end-start:.4f}, estimated train loss {loss:.4f}, acc {acc[0]*100:.4f}, {acc[1]*100:.4f}\n'
            if not args.cpu: torch.cuda.empty_cache()
            train_times.append(end-start)

            start = time.time()
            with torch.no_grad():
                model.eval()
                raw_preds = []
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
                loss_train = loss_fcn(raw_preds[:trainval_point], labels[train_nid]).item()
                loss_val = loss_fcn(raw_preds[trainval_point:valtest_point], labels[val_nid]).item()
                loss_test = loss_fcn(raw_preds[valtest_point:labeled_num_nodes], labels[test_nid]).item()

            if args.dataset != 'IMDB':
                preds = raw_preds.argmax(dim=-1)
            else:
                preds = (raw_preds > 0.).int()

            train_acc = evaluator(preds[:trainval_point], labels[train_nid])
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[val_nid])
            test_acc = evaluator(preds[valtest_point:labeled_num_nodes], labels[test_nid])

            end = time.time()
            log += f'evaluation Time: {end-start:.4f}, Train loss: {loss_train:.4f}, Val loss: {loss_val:.4f}, Test loss: {loss_test:.4f}\n'
            log += f'Train acc: ({train_acc[0]*100:.4f}, {train_acc[1]*100:.4f}), Val acc: ({val_acc[0]*100:.4f}, {val_acc[1]*100:.4f})'
            log += f', Test acc: ({test_acc[0]*100:.4f}, {test_acc[1]*100:.4f})\n'

            if (args.dataset != 'Freebase' and loss_val < best_val_loss) or (args.dataset == 'Freebase' and sum(val_acc) > sum(best_val)):
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

        all_pred = torch.empty((num_nodes, num_classes))
        all_pred[labeled_nid] = best_pred
        if len(full_loader):
            model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
            if not args.cpu: torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                raw_preds = []
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

            all_pred[extra_nid] = raw_preds
        torch.save(all_pred, f'{checkpt_file}.pt')

        if args.dataset != 'IMDB':
            predict_prob = all_pred.softmax(dim=1)
        else:
            predict_prob = torch.sigmoid(all_pred)

        test_logits = predict_prob[test_nid]
        if args.dataset != 'IMDB':
            pred = test_logits.cpu().numpy().argmax(axis=1)
            dl.gen_file_for_evaluate(test_idx=test_nid, label=pred, file_name=f"{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt")
        else:
            pred = (test_logits.cpu().numpy()>0.5).astype(int)
            dl.gen_file_for_evaluate(test_idx=test_nid, label=pred, file_name=f"{args.dataset}_{args.seed}_{checkpt_file.split('/')[-1]}.txt", mode='multi')

        if args.dataset != 'IMDB':
            preds = predict_prob.argmax(dim=1, keepdim=True)
        else:
            preds = (predict_prob > 0.5).int()
        train_acc = evaluator(preds[train_nid], labels[train_nid])
        val_acc = evaluator(preds[val_nid], labels[val_nid])
        test_acc = evaluator(preds[test_nid], labels[test_nid])

        print(f'train_acc ({train_acc[0]*100:.2f}, {train_acc[1]*100:.2f}) ' \
            + f'val_acc ({val_acc[0]*100:.2f}, {val_acc[1]*100:.2f}) ' \
            + f'test_acc ({test_acc[0]*100:.2f}, {test_acc[1]*100:.2f})')
        print(checkpt_file.split('/')[-1])

        del model
        if not args.cpu: torch.cuda.empty_cache()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument('--seeds', nargs='+', type=int, default=[1],
                        help='the seed used in the training')
    parser.add_argument('--dataset', type=str, default='DBLP',
                        choices=['DBLP', 'ACM', 'IMDB', 'Freebase'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--root', type=str, default='../data/')
    parser.add_argument('--epoch', type=int, default=50, help='Maxinum number of epochs.')
    parser.add_argument('--embed-size', type=int, default=256,
                        help='inital embedding size of nodes with no attributes')
    parser.add_argument('--num-hops', type=int, default=2,
                        help='number of hops for propagation of raw labels')
    parser.add_argument('--label-feats', action='store_true', default=False,
                        help='whether to use the label propagated features')
    parser.add_argument('--num-label-hops', type=int, default=2,
                        help='number of hops for propagation of raw features')
    ## For network structure
    parser.add_argument('--n-fp-layers', type=int, default=2,
                        help='the number of mlp layers for feature projection')
    parser.add_argument('--n-task-layers', type=int, default=3,
                        help='the number of mlp layers for the downstream task')
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout on activation')
    parser.add_argument('--input-drop', type=float, default=0.1,
                        help='input dropout of input features')
    parser.add_argument('--att-drop', type=float, default=0.,
                        help='attention dropout of model')
    parser.add_argument('--act', type=str, default='none',
                        choices=['none', 'relu', 'leaky_relu', 'sigmoid'],
                        help='the activation function of the transformer part')
    parser.add_argument('--residual', action='store_true', default=False,
                        help='whether to add residual branch the raw input features')
    ## for training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='whether to amp to accelerate training with float16(half) calculation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=50,
                        help='early stop patience')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'ACM':
        # We do not keep the node type `field`(F) as raw features represent exactly the distribution of nearby field nodes
        args.ACM_keep_F = False

    print(args)
    main(args)
