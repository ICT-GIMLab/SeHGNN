import torch
from sklearn.metrics import f1_score
import dgl
from utils import load_data, EarlyStopping, set_random_seed, setup_log_dir
from utils import load_dblp, load_acm, load_acm_hgb
import torch.nn.functional as F
from model_hetero import HAN, HAN_freebase
import argparse
import datetime


def score(logits, labels, bce=False):
    if bce:
        prediction = (logits > 0.).data.cpu().long().numpy()
    else:
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func, bce=False):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask], bce=bce)

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    if args.dataset == 'DBLP':
        g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
            val_mask, test_mask = load_dblp(args.tree_like_a_leaves, args.star_like_a_leaves)
        if args.tree_like_a_leaves or args.star_like_a_leaves:
            args.tgt_type = 'A'
        else:
            args.tgt_type = 'a'
    elif args.dataset == 'ACM':
        g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
            val_mask, test_mask = load_acm()
        args.tgt_type = 'p'
    elif args.dataset == 'ACM_HGB':
        g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
            val_mask, test_mask = load_acm_hgb(pp_symmetric=True)
        args.tgt_type = 'p'
    elif args.dataset == 'IMDB':
        g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
            val_mask, test_mask = load_imdb()
        args.tgt_type = 'm'
    else:
        assert 0, f'Not ready for dataset {args.dataset}'
    print(g)

    if len(args.metapaths):
        meta_paths = []
        for ele in args.metapaths:
            if len(ele) == 1:
                meta_paths.append([ele])
            else:
                ele = ele[::-1]
                meta_paths.append([
                    ele[i:i+2] for i in range(len(ele)-1)])
    elif args.dataset == 'DBLP': # apa, aptpa, apcpa
        '''
            a(0*) -- p(1) -- c(2)
                     |
                     t(3)
            a(0): [4057, 334]
            p(1): [18385, 4231]
            c(2): [26108, 50]
            t(3): None or [20, 20]
        ''' # PS: edge ap means a->p; metapath [['ap', 'pc']] means  a->p->c
        meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    elif args.dataset == 'ACM': # pap, pfp
        '''
            a -- p* -- f
            p: [4025, 1903]
            a: [17351, 1903]
            f: [72, 1903]
        '''
        meta_paths = [['pa', 'ap'], ['pf', 'fp']]
    elif args.dataset == 'ACM_HGB': # ppsp, pap, psp, ptp
        '''
            a(0) -- p(1*) -- s(2)
                    |
                    t(3)
            p: [5959, 1902]
            a: [3025, 1902]
            s: [56, 1902]
            t: None or [1902, 1902]
        '''
        meta_paths = [['pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    else:
        assert False, f'Unrecognized dataset {args.dataset}'

    if args.use_gat: # GAT requires the src node type and the tgt node type must be same, while GCN not
        assert False, 'not implement'
        # for metapath in metapaths:
        #     assert metapath[0] == args.tgt_type and metapath[1] == args.tgt_type

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = {k: v.to(args.device) for k, v in features.items()}
    labels = labels.to(args.device)
    train_mask = train_mask.to(args.device)
    val_mask = val_mask.to(args.device)
    test_mask = test_mask.to(args.device)
    g = g.to(args.device)

    if args.dataset == 'Freebase':
        model = HAN_freebase(
            meta_paths=meta_paths,
            in_size=features.shape[1],
            hidden_size=args.hidden_units,
            out_size=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout).to(args.device)
    else:
        model = HAN(
            meta_paths=meta_paths, tgt_type=args.tgt_type,
            in_sizes={k: v.shape[1] for k, v in features.items()},
            hidden_size=args.hidden_units,
            out_size=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout, use_gat=args.use_gat, attn_dropout=args.attn_dropout, residual=args.residual).to(args.device)
    print(model)

    stopper = EarlyStopping(patience=args.patience)
    if args.dataset == 'IMDB':
        loss_fcn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_times = []
    for epoch in range(args.num_epochs):
        if epoch > 5:
            torch.cuda.synchronize()
            tic = datetime.datetime.now()
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch > 5:
            torch.cuda.synchronize()
            toc = datetime.datetime.now()
            train_times.append(toc-tic)

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask], bce=args.dataset=='IMDB')
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn, bce=args.dataset=='IMDB')
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    total_times = datetime.timedelta(0)
    for ele in train_times:
        total_times += ele
    total_times = total_times / len(train_times)
    print('average train times', total_times)

    stopper.load_checkpoint(model)

    val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn, bce=args.dataset=='IMDB')
    print('Val loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        val_loss.item(), val_micro_f1, val_macro_f1))
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn, bce=args.dataset=='IMDB')
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seeds', nargs='+', type=int, default=[1],
                        help='Random seeds')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='DBLP',
                        choices=['DBLP', 'ACM', 'ACM_HGB', 'Freebase', 'IMDB'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument("--use-gat", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-heads", nargs='+',type=int, default=[8])
    parser.add_argument("--hidden-units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--attn-dropout", type=float, default=0.6)
    parser.add_argument("--weight-decay", type=float, default=0.001)

    parser.add_argument('--metapaths', nargs='+',type=str, default=[])
    parser.add_argument("--residual", action='store_true', default=False)

    parser.add_argument("--tree-like-a-leaves", action='store_true', default=False)
    parser.add_argument("--tree-like-a-roots", action='store_true', default=False)
    parser.add_argument("--star-like-a-leaves", action='store_true', default=False)
    parser.add_argument("--star-like-m-roots", action='store_true', default=False)

    args = parser.parse_args()

    # args = parser.parse_args('--dataset DBLP'.split(' '))
    # args = parser.parse_args('--dataset DBLP --metapaths a ap apc --num-heads 8'.split(' '))
    # args = parser.parse_args('--dataset IMDB --metapaths m ma md --num-heads 8'.split(' '))
    # args = parser.parse_args('--dataset DBLP --metapaths Ap Apa Apc --num-heads 8 --star-like-a-leaves'.split(' '))
    args = parser.parse_args('--dataset DBLP --metapaths AP APa APc APcp APcpa --num-heads 8 --tree-like-a-leaves'.split(' '))
    # set_random_seed(args.seed[0])

    print(args)
    args.log_dir = setup_log_dir(args)
    for seed in args.seeds:
        if seed > 0:
            set_random_seed(seed)
        main(args)
