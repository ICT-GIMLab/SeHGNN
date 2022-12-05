import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, independent_attn=False):
        super(SemanticAttention, self).__init__()
        self.independent_attn = independent_attn

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z): # z: (N, M, D)
        w = self.project(z) # (N, M, 1)
        if not self.independent_attn:
            w = w.mean(0, keepdim=True)  # (1, M, 1)
        beta = torch.softmax(w, dim=1)  # (N, M, 1) or (1, M, 1)

        return (beta * z).sum(1)  # (N, M, D)


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class SeHGNN(nn.Module):
    def __init__(self, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_dropout, label_drop,
                 n_layers_1, n_layers_2, act,
                 residual=False, bns=False, data_size=None, drop_metapath=0., num_heads=1,
                 remove_transformer=True, independent_attn=False):
        super(SeHGNN, self).__init__()
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(self.feat_keys) + len(self.label_feat_keys)
        self.tgt_type = tgt_type
        self.residual = residual
        self.remove_transformer = remove_transformer

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, nfeat).uniform_(-0.5, 0.5))
        else:
            self.labels_embeding = {}

        self.layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, num_channels, hidden, bias=True, cformat='channel-last'),
            # nn.LayerNorm([num_channels, hidden]),
            # nn.PReLU(),
            Conv1d1x1(hidden, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, 4, 1, bias=True, cformat='channel-last'),
            # nn.LayerNorm([4, hidden]),
            # nn.PReLU(),
        )

        if self.remove_transformer:
            self.layer_mid = SemanticAttention(hidden, hidden, independent_attn=independent_attn)
            self.layer_final = nn.Linear(hidden, hidden)
        else:
            self.layer_mid = Transformer(hidden, num_heads=num_heads)
            # self.layer_mid = Transformer(hidden, drop_metapath=drop_metapath)
            self.layer_final = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            return [
                nn.BatchNorm1d(hidden, affine=bns, track_running_stats=bns),
                nn.PReLU(),
                nn.Dropout(dropout)
            ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
            # [Dense(hidden, hidden, bns)] + add_nonlinear_layers(hidden, dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)]))
        # self.lr_output = FeedForwardNetII(
        #     hidden, hidden, nclass, n_layers_2, dropout, 0.5, bns)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)

        self.reset_parameters()

    def reset_parameters(self):
        # for k, v in self.embeding:
        #     v.data.uniform_(-0.5, 0.5)
        # for k, v in self.labels_embeding:
        #     v.data.uniform_(-0.5, 0.5)

        for layer in self.layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.layer_final.weight, gain=gain)
        nn.init.zeros_(self.layer_final.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, batch, feature_dict, label_dict={}, mask=None):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            mapped_feats = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            mapped_feats = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}
        else:
            assert 0

        mapped_label_feats = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        features = [mapped_feats[k] for k in self.feat_keys] + [mapped_label_feats[k] for k in self.label_feat_keys]

        # if mask is not None:
        #     if self.keys_sort:
        #         mask = [mask[k] for k in sorted_f_keys]
        #     else:
        #         mask = list(mask.values())
        #     mask = torch.stack(mask, dim=1)

        #     if len(label_list):
        #         if mask is not None:
        #             mask = torch.cat((mask, torch.ones((len(mask), len(label_list))).to(device=mask.device)), dim=1)

        B = num_node = mapped_feats[self.tgt_type].shape[0]
        C = self.num_channels
        D = mapped_feats[self.tgt_type].shape[1]

        features = torch.stack(features, dim=1) # [B, C, D]

        features = self.layers(features)
        if self.remove_transformer:
            features = self.layer_mid(features)
        else:
            features = self.layer_mid(features, mask=None).transpose(1,2)
        out = self.layer_final(features.reshape(B, -1))

        if self.residual:
            out = out + self.res_fc(mapped_feats[self.tgt_type])
        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)
        return out


from torch_geometric.nn import GATConv
class SeHGNN_NA(nn.Module):
    def __init__(self, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_dropout, label_drop,
                 n_layers_1, n_layers_2, act,
                 residual=False, bns=False, data_size=None, drop_metapath=0., num_heads=1):
        super(SeHGNN_NA, self).__init__()
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(self.feat_keys) + 1 + len(self.label_feat_keys)
        self.tgt_type = tgt_type
        self.residual = residual

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, nfeat).uniform_(-0.5, 0.5))
        else:
            self.labels_embeding = {}

        self.layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, num_channels, hidden, bias=True, cformat='channel-last'),
            # nn.LayerNorm([num_channels, hidden]),
            # nn.PReLU(),
            Conv1d1x1(hidden, hidden, num_channels, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, 4, 1, bias=True, cformat='channel-last'),
            # nn.LayerNorm([4, hidden]),
            # nn.PReLU(),
        )

        assert hidden % num_heads == 0, f'hidden {hidden} should be divisible by num_heads {num_heads}'
        self.feat_gat_layers = nn.ModuleDict({
            k: GATConv(hidden, hidden//num_heads, heads=num_heads, dropout=0.6) for k in self.feat_keys})

        self.layer_mid = Transformer(hidden, num_heads=1)
        # self.layer_mid = Transformer(hidden, drop_metapath=drop_metapath)
        self.layer_final = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            return [
                nn.BatchNorm1d(hidden, affine=bns, track_running_stats=bns),
                nn.PReLU(),
                nn.Dropout(dropout)
            ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
            # [Dense(hidden, hidden, bns)] + add_nonlinear_layers(hidden, dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)]))
        # self.lr_output = FeedForwardNetII(
        #     hidden, hidden, nclass, n_layers_2, dropout, 0.5, bns)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)

        self.reset_parameters()

    def reset_parameters(self):
        # for k, v in self.embeding:
        #     v.data.uniform_(-0.5, 0.5)
        # for k, v in self.labels_embeding:
        #     v.data.uniform_(-0.5, 0.5)

        for layer in self.layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        for k, v in self.feat_gat_layers.items():
            v.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.layer_final.weight, gain=gain)
        nn.init.zeros_(self.layer_final.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, batch, feat_adjs, label_dict={}, mask=None):
        mapped_feats = {str(k): self.input_drop(x @ self.embeding[str(k)]) for k, x in self.feats.items()}

        aggregated_feats = {}
        for k in self.feat_gat_layers.keys():
            tgt, src = k[0], k[-1]
            assert len(k) > 1 and tgt == self.tgt_type
            aggregated_feats[k] = self.feat_gat_layers[k]((
                mapped_feats[src], mapped_feats[self.tgt_type][batch]), feat_adjs[k])
        aggregated_feats[self.tgt_type] = mapped_feats[self.tgt_type][batch]

        mapped_label_feats = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        features = [aggregated_feats[k] for k in self.feat_keys] + [aggregated_feats[self.tgt_type]] + [mapped_label_feats[k] for k in self.label_feat_keys]

        # if mask is not None:
        #     if self.keys_sort:
        #         mask = [mask[k] for k in sorted_f_keys]
        #     else:
        #         mask = list(mask.values())
        #     mask = torch.stack(mask, dim=1)

        B = num_node = len(batch)
        C = self.num_channels
        D = aggregated_feats[self.tgt_type].shape[1]

        features = torch.stack(features, dim=1) # [B, C, D]

        features = self.layers(features)
        features = self.layer_mid(features, mask=None).transpose(1,2)
        out = self.layer_final(features.reshape(B, -1))

        if self.residual:
            out = out + self.res_fc(aggregated_feats[self.tgt_type])
        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)
        return out


class SeHGNN_block(nn.Module):
    def __init__(self, nfeat, hidden, feat_keys, label_feat_keys, tgt_type, dropout, num_fp_layers=2):
        super(SeHGNN_block, self).__init__()
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.tgt_type = tgt_type
        self.num_channels = num_channels = len(self.feat_keys) + len(self.label_feat_keys)

        def add_fp_layers(cin, cout):
            return [
                Conv1d1x1(cin, cout, num_channels, bias=True, cformat='channel-first'),
                nn.LayerNorm([num_channels, cout]),
                nn.PReLU(),
                nn.Dropout(dropout)
            ]

        layers = add_fp_layers(nfeat, hidden)
        for i in range(1, num_fp_layers):
            layers += add_fp_layers(hidden, hidden)
        self.layers = nn.Sequential(*layers)

        self.layer_mid = Transformer(hidden, num_heads=1)
        # self.layer_mid = Transformer(hidden, drop_metapath=drop_metapath)
        self.layer_final = nn.Linear(num_channels * hidden, hidden)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        self.layer_mid.reset_parameters()
        nn.init.xavier_uniform_(self.layer_final.weight, gain=gain)
        nn.init.zeros_(self.layer_final.bias)

    def forward(self, x, label_dict={}, mask=None):
        x = torch.stack(
            [x[k] for k in self.feat_keys] + [label_dict[k] for k in self.label_feat_keys], dim=1)
        B, C, D = x.size()
        assert C == self.num_channels

        x = self.layers(x)
        x = self.layer_mid(x, mask=mask).transpose(1,2)
        return self.layer_final(x.reshape(B, -1))


class SeHGNN_2L(nn.Module):
    def __init__(self, nfeat, hidden, nclass, feat_keys1, feat_keys2, label_feat_keys, node_types,
                 dropout, input_drop, att_dropout, label_drop,
                 n_layers_1, n_layers_2, act,
                 residual=False, bns=False, data_size=None, drop_metapath=0., num_heads=1):
        super(SeHGNN_2L, self).__init__()
        self.label_feat_keys = sorted(label_feat_keys)
        self.node_types = node_types
        self.tgt_type = self.node_types[0]
        self.residual = residual

        self.feat_keys1 = {}
        for tgt in self.node_types:
            self.feat_keys1[tgt] = [k for k in feat_keys1 if k[0] == tgt] + [tgt]

        self.layer1_blocks = nn.ModuleDict({
            tgt: SeHGNN_block(nfeat, hidden, self.feat_keys1[tgt], {}, tgt, dropout) for tgt in self.node_types})

        self.feat_keys2 = sorted(feat_keys2) + [self.tgt_type]

        self.layer2 = SeHGNN_block(hidden, hidden, self.feat_keys2, self.label_feat_keys, self.tgt_type, dropout)

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, nfeat).uniform_(-0.5, 0.5))
        else:
            self.labels_embeding = {}

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            return [
                nn.BatchNorm1d(hidden, affine=bns, track_running_stats=bns),
                nn.PReLU(),
                nn.Dropout(dropout)
            ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
            # [Dense(hidden, hidden, bns)] + add_nonlinear_layers(hidden, dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)]))
        # self.lr_output = FeedForwardNetII(
        #     hidden, hidden, nclass, n_layers_2, dropout, 0.5, bns)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)

        self.reset_parameters()

    def reset_parameters(self):
        # for k, v in self.embeding:
        #     v.data.uniform_(-0.5, 0.5)
        # for k, v in self.labels_embeding:
        #     v.data.uniform_(-0.5, 0.5)

        for layer in self.layer1_blocks.values():
            layer.reset_parameters()
        self.layer2.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, layer1_feats, batch1, layer2_feats, batch2, label_dict={}):
        mapped_raw_feats = {str(k): self.input_drop(x @ self.embeding[str(k)]) for k, x in self.feats.items()}

        aggregated_feats = {k: v @ mapped_raw_feats[k[-1]] for k, v in layer1_feats.items()}
        for k, batch in batch1.items():
            aggregated_feats[k] = mapped_raw_feats[k][batch]

        features = {}
        for tgt in batch1.keys():
            features[tgt] = self.layer1_blocks[tgt]({k: v for k, v in aggregated_feats.items() if k[0] == tgt}, {})

        aggregated_feats = {k: v @ features[k[-1]] for k, v in layer2_feats.items()}
        aggregated_feats[self.tgt_type] = mapped_raw_feats[self.tgt_type][batch2]

        mapped_label_feats = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        out = self.layer2(aggregated_feats, mapped_label_feats)

        if self.residual:
            out = out + self.res_fc(aggregated_feats[self.tgt_type])
        out = self.dropout(self.prelu(out))
        out = self.lr_output(out)
        return out
