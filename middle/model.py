import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, act='none', drop_metapath=0.):
        super(Transformer, self).__init__()
        self.query ,self.key, self.value = [self._conv(n_channels, c) for c in (n_channels//8, n_channels//8, n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))
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
        self.drop_metapath = drop_metapath

    def _conv(self,n_in, n_out):
        return torch.nn.utils.spectral_norm(nn.Conv1d(n_in, n_out, 1, bias=False))

    def forward(self, x, mask=None):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1) # [726, 512, 9]
        f, g, h = self.query(x), self.key(x), self.value(x)
        # f = F.dropout(self.query(x), training=self.training) # [726, 64, 9]
        # g = F.dropout(self.key(x), training=self.training)   # [726, 64, 9]
        # h = F.dropout(self.value(x), training=self.training) # [726, 512, 9]
        beta = F.softmax(self.act(torch.bmm(f.transpose(1,2), g)), dim=1)
        if self.drop_metapath > 0 and self.training:
            if mask is not None:
                print('Warning: drop_metapath is not used as mask is given')
            else:
                with torch.no_grad():
                    mask = F.dropout(torch.ones(beta.size(0), beta.size(1)), p=self.drop_metapath).to(device=x.device)
        if mask is not None:
            beta = beta * mask.unsqueeze(-1)
            beta = beta / (beta.sum(dim=1, keepdim=True) + 1e-6)
            o = self.gamma * torch.bmm(h, beta) * mask.unsqueeze(1) + x
        else:
            o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


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
    def __init__(self, nfeat, hidden, nclass, num_channels,
                 dropout, input_drop, att_dropout, label_drop,
                 n_layers_1, n_layers_2, n_layers_3, act,
                 residual=False, bns=False,
                 data_size=None, embed_train=False, label_names=None, drop_metapath=0.):
        super(SeHGNN, self).__init__()
        self.num_channels = num_channels
        self.residual = residual

        # self.data_size = data_size
        if data_size is not None:
            if isinstance(data_size, list):
                assert 0
                # self.embeding = nn.ParameterList([nn.Parameter(
                #     torch.Tensor(cin, nfeat).uniform_(-0.5, 0.5)) for cin in data_size])
            elif isinstance(data_size, dict):
                self.embeding = nn.ParameterDict({})
                for k, v in data_size.items():
                    self.embeding[str(k)] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
            else:
                assert 0
        else:
            self.embeding = None
        self.embed_train = embed_train

        if len(label_names):
            self.labels_embeding = nn.ParameterDict({})
            for k in label_names:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, nfeat).uniform_(-0.5, 0.5))
        else:
            self.labels_embeding = None

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
        self.layer_mid = Transformer(hidden)
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

        self.keys_sort = True
        self.reset_parameters()

    def reset_parameters(self):
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

    def forward(self, feature_list, label_list={}, mask=None):
        if self.keys_sort:
            sorted_f_keys = sorted(feature_list.keys())
            sorted_l_keys = sorted(label_list.keys())

        feature_hop0 = None
        if self.embeding is not None:
            if self.embed_train:
                if isinstance(self.embeding, nn.ParameterList):
                    feature_list = [x @ w for x, w in zip(feature_list, self.embeding)]
                    feature_hop0 = self.input_drop(feature_list[0])
                elif isinstance(self.embeding, nn.ParameterDict):
                    if len(self.embeding) == len(feature_list):
                        feature_list = {k: x @ self.embeding[k] for k, x in feature_list.items()}
                    else:
                        feature_list = {k: x @ self.embeding[k[-1]] for k, x in feature_list.items()}
                    if self.keys_sort:
                        feature_list = [feature_list[k] for k in sorted_f_keys]
                        if self.residual:
                            if 0 in sorted_f_keys:
                                pos = np.where(np.array(feature_list) == 0)[0][0]
                            else:
                                name_lens = np.array([len(k) for k in sorted_f_keys])
                                assert np.sum(name_lens == 1) == 1
                                pos = np.where(name_lens == 1)[0][0]
                            feature_hop0 = self.input_drop(feature_list[pos])
                    else:
                        feature_list = list(feature_list.values())
                else:
                    assert False
            else:
                with torch.no_grad():
                    if isinstance(self.embeding, nn.ParameterList):
                        feature_list = [x @ w for x, w in zip(feature_list, self.embeding)]
                        feature_hop0 = self.input_drop(feature_list[0])
                    elif isinstance(self.embeding, nn.ParameterDict):
                        if len(self.embeding) == len(feature_list):
                            feature_list = {k: x @ self.embeding[k] for k, x in feature_list.items()}
                        else:
                            feature_list = {k: x @ self.embeding[k[-1]] for k, x in feature_list.items()}
                        if self.keys_sort:
                            feature_list = [feature_list[k] for k in sorted_f_keys]
                            if self.residual:
                                if 0 in sorted_f_keys:
                                    pos = np.where(np.array(feature_list) == 0)[0][0]
                                else:
                                    name_lens = np.array([len(k) for k in sorted_f_keys])
                                    assert np.sum(name_lens == 1) == 1
                                    pos = np.where(name_lens == 1)[0][0]
                                feature_hop0 = self.input_drop(feature_list[pos])
                        else:
                            feature_list = list(feature_list.values())
                    else:
                        assert False
        if mask is not None:
            if self.keys_sort:
                mask = [mask[k] for k in sorted_f_keys]
            else:
                mask = list(mask.values())
            mask = torch.stack(mask, dim=1)

        if len(label_list):
            label_list = {k: x @ self.labels_embeding[k] for k, x in label_list.items()}
            if self.keys_sort:
                feature_list += [label_list[k] for k in sorted_l_keys]
            else:
                feature_list += list(label_list.values())
            if mask is not None:
                mask = torch.cat((mask, torch.ones((len(mask), len(label_list))).to(device=mask.device)), dim=1)

        B = num_node = feature_list[0].shape[0]
        C = num_channels = len(feature_list)
        D = feature_list[0].shape[1]

        feature_list = self.input_drop(torch.stack(feature_list, dim=1))
        right_1 = self.layers(feature_list)

        right_1 = self.layer_mid(right_1.transpose(1,2), mask=mask).transpose(1,2)
        right_1 = self.layer_final(right_1.reshape(B, -1))

        if self.residual:
            if feature_hop0 is None:
                print('Warning: no feature_hop0 generated', flush=True)
            else:
                right_1 += self.res_fc(feature_hop0)
        right_1 = self.dropout(self.prelu(right_1))
        # if self.pre_dropout:
        #     right_1=self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        # right_1 += self.label_fc(self.label_drop(label_emb))
        # torch.cuda.synchronize()
        # toc = datetime.datetime.now()
        # print(f'mul {toc-tic}', flush=True)
        return right_1
