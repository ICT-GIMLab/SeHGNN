import torch
from sklearn.metrics import f1_score
import dgl
from utils import load_data, EarlyStopping, set_random_seed, setup_log_dir
import torch.nn.functional as F
from model_hetero import HAN, HAN_freebase
import argparse
import datetime


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, meta_paths = load_data(args.dataset, feat_type=0)

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args.device)
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
            meta_paths=meta_paths,
            in_size=features.shape[1],
            hidden_size=args.hidden_units,
            out_size=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout, use_gat=args.use_gat, attn_dropout=args.attn_dropout).to(args.device)

    stopper = EarlyStopping(patience=args.patience)
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

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
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

    val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
    print('Val loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        val_loss.item(), val_micro_f1, val_macro_f1))
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', nargs='+', type=int, default=[1],
                        help='Random seeds')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='DBLP',
                        choices=['DBLP', 'ACM', 'ACM_HGB', 'Freebase'])
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

    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    print(args)
    for seed in args.seed:
        if seed > 0:
            set_random_seed(seed)
        args.log_dir = setup_log_dir(args)
        main(args)
