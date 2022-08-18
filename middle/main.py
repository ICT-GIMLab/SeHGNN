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


def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)

    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid \
        = load_dataset(args)
    test_nid_full = np.nonzero(dl.labels_test_full['mask'])[0]

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
        extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        extra_metapath = []
    elif args.dataset == 'Freebase':
        tgt_type = '0'
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

    # compute k-hop feature
    prop_tic = datetime.datetime.now()
    if args.dataset != 'Freebase':
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1
        g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, extra_metapath, echo=True)
        feats = {}
        keys = list(g.nodes[tgt_type].data.keys())
        print(f'For tgt {tgt_type}, Involved keys {keys}')
        for k in keys:
            feats[k] = g.nodes[tgt_type].data.pop(k)
    else:
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1
        save_name = f'Freebase_feat_adjs_seed{args.seed}_hop{args.num_hops}.pt'
        if args.seed > 0 and os.path.exists(save_name):
            meta_adjs = torch.load(save_name)
        else:
            meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device=prop_device)
            if args.seed > 0:
                torch.save(meta_adjs, save_name)
        feats = {k: v.clone() for k, v in meta_adjs.items() if len(k) <= args.num_hops + 1 or k in extra_metapath}
        if '0' not in feats:
            feats['0'] = SparseTensor.eye(dl.nodes['count'][0])
        print(f'For tgt {tgt_type}, Involved keys {feats.keys()}')

    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        data_size = {k: v.size(-1) for k, v in feats.items()}
        # weights = [torch.Tensor(ele.size(-1), args.embed_size).uniform_(-0.5, 0.5) for ele in feats]
        # feats = [x @ w for x, w in zip(feats, weights)]
        feats = {k: v[init2sort] for k, v in feats.items()}
    elif args.dataset == 'Freebase':
        data_size = dict(dl.nodes['count'])
        for k, v in tqdm(feats.items()):
            elements = k.split('<-')
            if elements[-1] == '0':
                # feats[k] = v[init2sort[:total_num_nodes], init2sort]
                # feats[k] = v[init2sort, init2sort]
                feats[k], _ = v.sample_adj(init2sort, -1, False) # faster, 50% time acceleration
            else:
                # feats[k] = v[init2sort[:total_num_nodes]]
                feats[k] = v[init2sort]
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

    for stage in range(args.start_stage, len(args.stages)):
        epochs = args.stages[stage]

        if len(args.reload):
            pt_path = f'output/ogbn-mag/{args.reload}_{stage-1}.pt'
            assert os.path.exists(pt_path)
            print(f'Reload raw_preds from {pt_path}', flush=True)
            raw_preds = torch.load(pt_path, map_location='cpu')
            if args.dataset != 'IMDB':
                predict_prob = raw_preds.softmax(dim=1)
            else:
                predict_prob = torch.sigmoid(raw_preds)

        # =======
        # Expand training set
        # =======
        if stage > 0:
            if args.dataset != 'IMDB':
                preds = predict_prob.argmax(dim=1, keepdim=True)
            else:
                preds = (predict_prob > 0.5).int()

            train_acc = evaluator(labels[:trainval_point], preds[:trainval_point])
            val_acc = evaluator(labels[trainval_point:valtest_point], preds[trainval_point:valtest_point])
            test_acc = evaluator(labels[valtest_point:total_num_nodes], preds[valtest_point:total_num_nodes])

            print(f'Stage {stage-1} history model: (Micro-F1, Macro-F1)\n\t' \
                + f'train_acc ({train_acc[0]*100:.2f}, {train_acc[1]*100:.2f}) ' \
                + f'val_acc ({val_acc[0]*100:.2f}, {val_acc[1]*100:.2f}) ' \
                + f'test_acc ({test_acc[0]*100:.2f}, {test_acc[1]*100:.2f})')

            if args.dataset != 'IMDB':
                confident_mask = predict_prob.max(1)[0] > args.threshold
                val_enhance_offset  = torch.where(confident_mask[trainval_point:valtest_point])[0]
                test_enhance_offset = torch.where(confident_mask[valtest_point:total_num_nodes])[0]
                val_enhance_nid     = val_enhance_offset + trainval_point
                test_enhance_nid    = test_enhance_offset + valtest_point
                enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

                extra_enhance_offset = torch.where(confident_mask[total_num_nodes:])[0]
                extra_enhance_id = extra_enhance_offset + total_num_nodes

                val_confident_level = (predict_prob[val_enhance_nid].argmax(1) == labels[val_enhance_nid]).sum() / len(val_enhance_nid)
                print(f'Stage: {stage}, confident nodes: {len(enhance_nid)} / {total_num_nodes - trainval_point}')
                print(f'\t\t val confident nodes: {len(val_enhance_nid)},  val confident level: {val_confident_level}')
                test_confident_level = (predict_prob[test_enhance_nid].argmax(1) == labels[test_enhance_nid]).sum() / len(test_enhance_nid)
                print(f'\t\ttest confident nodes: {len(test_enhance_nid)}, test confident_level: {test_confident_level}')
            else:
                confident_mask = torch.abs(predict_prob - 0.5) > args.threshold
                print(f'Stage: {stage}, confident points: {confident_mask[trainval_point:].sum()} / {(num_nodes - trainval_point) * predict_prob.size(1)}')
                val_confident_result = (preds[trainval_point:valtest_point] == labels[trainval_point:valtest_point])[confident_mask[trainval_point:valtest_point]]
                print(f'Stage: {stage}, val confident points: {confident_mask[trainval_point:valtest_point].sum()} / {(valtest_point - trainval_point) * predict_prob.size(1)}, '
                    + f'val confident level: {val_confident_result.sum() / confident_mask[trainval_point:valtest_point].sum()}')

                confident_mask = confident_mask.sum(1) > 3
                val_enhance_offset  = torch.where(confident_mask[trainval_point:valtest_point])[0]
                test_enhance_offset = torch.where(confident_mask[valtest_point:total_num_nodes])[0]
                val_enhance_nid     = val_enhance_offset + trainval_point
                test_enhance_nid    = test_enhance_offset + valtest_point
                enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        if args.label_feats:
            if args.dataset != 'IMDB':
                if stage > 0:
                    label_onehot = predict_prob.clone()
                    label_onehot[~confident_mask] = 0
                    label_onehot = label_onehot[sort2init]
                else:
                    label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = F.one_hot(init_labels[train_nid], num_classes).float()
            else:
                if stage > 0:
                    label_onehot = predict_prob * 2 - 1
                    # label_onehot[~confident_mask] = 0
                    label_onehot = label_onehot[sort2init]
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
            if stage == args.start_stage:
                meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)

            if args.dataset == 'Freebase':
                label_onehot_g = label_onehot.to(prop_device)
            for k, v in tqdm(meta_adjs.items()):
                if args.dataset != 'Freebase':
                    label_feats[k] = remove_diag(v) @ label_onehot
                else:
                    label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)
            if args.dataset == 'Freebase':
                del label_onehot_g
                torch.cuda.empty_cache()
            gc.collect()

            if args.dataset == 'Freebase':
                condition = lambda ra,rb,rc,k: rb > 0.2
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False)

                left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                remove_keys = list(set(list(label_feats.keys())) - set(left_keys))
                for k in remove_keys:
                    label_feats.pop(k)
            elif args.dataset == 'IMDB':
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
        if stage == 0:
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            if stage > args.start_stage:
                del train_loader
            train_batch_size = int(args.batch_size * len(train_nid) / (len(enhance_nid) + len(train_nid)))
            train_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)), batch_size=train_batch_size, shuffle=True, drop_last=False)
            enhance_batch_size = int(args.batch_size * len(enhance_nid) / (len(enhance_nid) + len(train_nid)))
            enhance_loader = torch.utils.data.DataLoader(
                enhance_nid, batch_size=enhance_batch_size, shuffle=True, drop_last=False)

            # teacher_probs = torch.zeros(predict_prob.shape[0], predict_prob.shape[1])
            # teacher_probs[enhance_nid,:] = predict_prob[enhance_nid,:]

        # =======
        # Mask & Smooth
        # =======
        with_mask = False
        if args.dataset == 'Freebase':
            init_mask = {k: v.storage.rowcount() != 0 for k, v in feats.items()}
            with_mask = True
        else:
            print(f'TODO: `with_mask` has not be implemented for {args.dataset}')
        with_smooth = False

        if with_mask:
            train_mask = {k: (v[:total_num_nodes] & (torch.randn(total_num_nodes) > 0)).float() for k, v in init_mask.items()}
            full_mask = {k: v.float() for k, v in init_mask.items()}
        else:
            train_mask = full_mask = None

        # Freebase train/val/test/full_nodes: 1909/477/5568/40402
        # IMDB     train/val/test/full_nodes: 1097/274/3202/359
        if stage > args.start_stage:
            del eval_loader, full_loader
        eval_loader, full_loader = [], []
        if args.dataset == 'Freebase':
            batchsize = 2 * args.batch_size
            for batch_idx in range((total_num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize
                batch_end = min(total_num_nodes, (batch_idx+1) * batchsize)

                if isinstance(feats, list):
                    batch_feats = [x[batch_start:batch_end] for x in feats]
                elif isinstance(feats, dict):
                    batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
                else:
                    assert 0
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
                if with_mask:
                    batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                else:
                    batch_mask = None
                eval_loader.append((batch_feats, batch_labels_feats, batch_mask))
            for batch_idx in range((num_nodes-total_num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize + total_num_nodes
                batch_end = min(num_nodes, (batch_idx+1) * batchsize + total_num_nodes)

                if isinstance(feats, list):
                    batch_feats = [x[batch_start:batch_end] for x in feats]
                elif isinstance(feats, dict):
                    batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
                else:
                    assert 0
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
                if with_mask:
                    batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                else:
                    batch_mask = None
                full_loader.append((batch_feats, batch_labels_feats, batch_mask))
        else:
            batchsize = args.batch_size
            for batch_idx in range((num_nodes-1) // batchsize + 1):
                batch_start = batch_idx * batchsize
                batch_end = min(num_nodes, (batch_idx+1) * batchsize)

                if isinstance(feats, list):
                    batch_feats = [x[batch_start:batch_end] for x in feats]
                elif isinstance(feats, dict):
                    batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
                else:
                    assert 0
                batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
                if with_mask:
                    batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                else:
                    batch_mask = None
                eval_loader.append((batch_feats, batch_labels_feats, batch_mask))

        # =======
        # Construct network
        # =======
        torch.cuda.empty_cache()
        gc.collect()
        model = SeHGNN(args.embed_size, args.hidden, num_classes, len(feats) + len(label_feats),
            args.dropout, args.input_drop, args.att_drop, args.label_drop,
            args.n_layers_1, args.n_layers_2, args.n_layers_3, args.act, args.residual, bns=args.bns,
            data_size=data_size, embed_train=args.embed_train, label_names=list(label_feats.keys()))
        model = model.to(device)
        if stage == args.start_stage:
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

        last_acc_val = (0, 0)
        last_best_epoch = -1
        train_times = []

        for epoch in tqdm(range(epochs)):
            gc.collect()
            torch.cuda.synchronize()
            start = time.time()
            if stage > 0:
                loss, acc = train_multi_stage(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, enhance_loader, evaluator, predict_prob, args.gama, mask=train_mask, scalar=scalar)
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
                for batch_feats, batch_labels_feats, batch_mask in eval_loader:
                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                    if with_mask:
                        batch_mask = {k: x.to(device) for k, x in batch_mask.items()}
                    else:
                        batch_mask = None
                    raw_preds.append(model(batch_feats, batch_labels_feats, batch_mask).cpu())

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

            last_acc_val = val_acc

            end = time.time()
            log += f'evaluation Time: {end-start}, Train loss: {loss_train}, Val loss: {loss_val}, Test loss: {loss_test}\n'
            log += 'Train acc: ({:.4f}, {:.4f}), Val acc: ({:.4f}, {:.4f}), Test acc: ({:.4f}, {:.4f}) ({})\n'.format(
                train_acc[0]*100, train_acc[1]*100, val_acc[0]*100, val_acc[1]*100, test_acc[0]*100, test_acc[1]*100, total_num_nodes-valtest_point)

            if args.store_model:
                torch.save(model.state_dict(), f'{checkpt_file}_{stage}_{epoch}.pkl')

            val_loss_list.append(loss_val)
            test_loss_list.append(loss_test)
            store_list.append(raw_preds)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

            if len(val_loss_list) > args.moving_k:
                assert len(val_loss_list) == args.moving_k + 1
                val_loss_list.pop(0)
                test_loss_list.pop(0)
                val_acc_list.pop(0)
                test_acc_list.pop(0)
                store_list.pop(0)
            gc.collect()

            if len(val_acc_list) >= args.moving_k:
                if args.moving_k == 1:
                    val_acc0, val_acc1 = val_acc_list[0]
                    test_acc0, test_acc1 = test_acc_list[0]
                    current_val_loss = val_loss_list[0]
                    current_test_loss = test_loss_list[0]
                else:
                    val_acc0, val_acc1 = np.mean(val_acc_list, axis=0)
                    test_acc0, test_acc1 = np.mean(test_acc_list, axis=0)
                    current_val_loss = sum(val_loss_list) / args.moving_k
                    current_test_loss = sum(test_loss_list) / args.moving_k

                if (args.dataset != 'Freebase' and current_val_loss <= best_val_loss) or (args.dataset == 'Freebase' and val_acc0 + val_acc1 >= sum(best_val)):
                    best_epoch = epoch
                    best_val_loss = current_val_loss
                    best_test_loss = current_test_loss
                    best_val = (val_acc0, val_acc1)
                    best_test = (test_acc0, test_acc1)
                    last_best_epoch = epoch

                    best_pred = store_list[0]
                    for pred in store_list[1:]:
                        best_pred = best_pred + pred
                    best_pred = best_pred / args.moving_k

            if last_best_epoch > 0 and epoch - last_best_epoch > args.patience: break

            if (epoch+1) % 10 == 0:
                log += "Best Epoch {}, Val loss {:.4f} ({:.4f}, {:.4f}), Test loss {:.4f} ({:.4f}, {:.4f})\n".format(
                    best_epoch, best_val_loss, best_val[0]*100, best_val[1]*100, best_test_loss, best_test[0]*100, best_test[1]*100)

                if args.store_model and best_epoch >= 0:
                    for filename in os.listdir(checkpt_folder):
                        epochs = re.findall(f'{checkpt_file.split("/")[-1]}_{stage}_([0-9]*).pkl', filename)
                        if len(epochs) > 0:
                            param_epoch = int(epochs[0])
                            if param_epoch + args.moving_k <= best_epoch or \
                              (best_epoch < param_epoch and param_epoch + args.moving_k <= epoch):
                                print(f'remove {checkpt_folder}/{filename}')
                                os.remove(f'{checkpt_folder}/{filename}')
            print(log)

        print('average train times', sum(train_times) / len(train_times))

        print("Best Epoch {}, Val loss {:.4f} ({:.4f}, {:.4f}), Test loss {:.4f} ({:.4f}, {:.4f})".format(
            best_epoch, best_val_loss, best_val[0]*100, best_val[1]*100, best_test_loss, best_test[0]*100, best_test[1]*100))
        torch.save(best_pred, f'{checkpt_file}_{stage}.pt')

        if args.dataset != 'IMDB':
            predict_prob = best_pred.softmax(dim=1)
        else:
            predict_prob = torch.sigmoid(best_pred)

        if args.store_model:
            del store_list
            store_list = []
            rerun_epoch_list = []
            for filename in os.listdir(checkpt_folder):
                epochs = re.findall(f'{checkpt_file.split("/")[-1]}_{stage}_([0-9]*).pkl', filename)
                if len(epochs) > 0:
                    epoch = int(epochs[0])
                    if epoch <= best_epoch and epoch + args.moving_k > best_epoch:
                        if args.dataset == 'Freebase':
                            print('Re-run', epoch, filename)
                            rerun_epoch_list.append(epoch)
                            model.load_state_dict(torch.load(f'{checkpt_folder}/{filename}', map_location='cpu'), strict=True)
                            torch.cuda.empty_cache()
                            with torch.no_grad():
                                model.eval()
                                assert with_smooth == False
                                raw_preds = []
                                for batch_feats, batch_labels_feats, batch_mask in full_loader:
                                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                                    if with_mask:
                                        batch_mask = {k: x.to(device) for k, x in batch_mask.items()}
                                    else:
                                        batch_mask = None
                                    raw_preds.append(model(batch_feats, batch_labels_feats, batch_mask).cpu())
                            pred = torch.cat(raw_preds, dim=0)
                            store_list.append(pred)
                        os.remove(f'{checkpt_folder}/{filename}')
                    else:
                        print(f'remove {checkpt_folder}/{filename}')
                        os.remove(f'{checkpt_folder}/{filename}')

            if args.dataset == 'Freebase':
                assert len(rerun_epoch_list) > 0
                print(f'Re-run following epochs:\n\t{rerun_epoch_list}')

                best_pred = store_list[0]
                for pred in store_list[1:]:
                    best_pred = best_pred + pred
                best_pred = best_pred / len(store_list)

                predict_prob = torch.cat((predict_prob, best_pred.softmax(dim=1)), dim=0)

        test_logits = predict_prob[sort2init][test_nid_full]
        if args.dataset != 'IMDB':
            pred = test_logits.cpu().numpy().argmax(axis=1)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred, file_path=f"{args.dataset}_{args.seed}_{stage}_{checkpt_file.split('/')[-1]}.txt")
        else:
            pred = (test_logits.cpu().numpy()>0.5).astype(int)
            dl.gen_file_for_evaluate(test_idx=test_nid_full, label=pred, file_path=f"{args.dataset}_{args.seed}_{stage}_{checkpt_file.split('/')[-1]}.txt", mode='multi')

    if args.dataset != 'IMDB':
        preds = predict_prob.argmax(dim=1, keepdim=True)
    else:
        preds = (predict_prob > 0.5).int()
    train_acc = evaluator(labels[:trainval_point], preds[:trainval_point])
    val_acc = evaluator(labels[trainval_point:valtest_point], preds[trainval_point:valtest_point])
    test_acc = evaluator(labels[valtest_point:total_num_nodes], preds[valtest_point:total_num_nodes])

    print(f'Stage {stage} history model:\n\t' \
        + f'train_acc ({train_acc[0]*100:.2f}, {train_acc[1]*100:.2f}) ' \
        + f'val_acc ({val_acc[0]*100:.2f}, {val_acc[1]*100:.2f}) ' \
        + f'test_acc ({test_acc[0]*100:.2f}, {test_acc[1]*100:.2f})')
    print(checkpt_file.split('/')[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default="../data/")
    parser.add_argument("--stages", nargs='+',type=int, default=[300, 300],
                        help="The epoch setting for each stage.")
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--embed-train", action='store_true', default=False,
                        help="whether to use train embeddings")
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
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of layers of residual label connection")
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
    # parser.add_argument("--use-emb", type=str)
    # parser.add_argument("--alpha", type=float, default=0.5,
    #                     help="initial residual parameter for the model")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="the threshold of multi-stage learning, confident nodes "
                           + "whose score above this threshold would be added into the training set")
    parser.add_argument("--gama", type=float, default=0.5,
                        help="parameter for the KL loss")
    # parser.add_argument("--involve-val-labels", action='store_true', default=False,
    #                     help="whether to process the input features")
    # parser.add_argument("--involve-extra-labels", action='store_true', default=False,
    #                     help="whether to process the input features")
    parser.add_argument("--drop-metapath", type=float, default=0,
                        help="whether to process the input features")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--reload", type=str, default='')
    parser.add_argument("--moving-k", type=int, default=10)
    parser.add_argument("--store-model", action='store_true', default=False,
                        help="whether to save model per epoch per stage. WARNING: it costs lots of disk")

    args = parser.parse_args()

    args.bns = args.bns and args.dataset == 'Freebase' # remove bn for full-batch learning
    if args.dataset == 'ACM':
        args.ACM_keep_F = False
    args.seed = args.seeds[0]
    print(args)
    # main(args)

    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        main(args)