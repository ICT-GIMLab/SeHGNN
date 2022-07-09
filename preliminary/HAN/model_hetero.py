import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv, GraphConv


echo_time = True

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)


def edge_dropout(g, weight, p, training=True):
    if training == False or p >= 1:
        return torch.ones_like(weight)

    with torch.no_grad():
        ones = torch.ones_like(weight)
        mask = F.dropout(ones, p=p, training=training)

        g.edata.update({'mask': torch.cat((ones, mask), dim=1)})
        g.update_all(fn.copy_e('mask', 'm'),
                     fn.sum('m', 'num_msgs'))

        num_msgs = g.dstdata.pop('num_msgs')
        num_msgs = num_msgs[:,0] / num_msgs[:,1]
        num_msgs[torch.isnan(num_msgs) | torch.isinf(num_msgs)] = -1
        g.dstdata.update({'num_msgs': num_msgs})

        g.apply_edges(lambda edges: {'mask_value': edges.dst['num_msgs']})
        mask_value = g.edata.pop('mask_value').reshape(weight.shape)

        final_mask = mask * mask_value * (mask_value > 0).float() + (mask_value < 0).float()

        # g.edata.update({'mask': weight * final_mask})
        # g.update_all(fn.copy_e('mask', 'm'),
        #              fn.sum('m', 'num_msgs'))
        # num_msgs = g.dstdata.pop('num_msgs')
        # # func(num_msgs)
        # print(torch.unique(num_msgs, return_counts=True), flush=True)

        g.edata.pop('mask')

    return final_mask



class MeanAggregator(nn.Module):
    def __init__(self, in_feats, out_feats, feat_drop, aggr_drop, norm='right',
                 weight=True, bias=True, activation=None, allow_zero_in_degree=False):
        super(MeanAggregator, self).__init__()
        self.feat_drop = feat_drop
        self.aggr_drop = aggr_drop
        self.gcnconv = GraphConv(in_feats, out_feats, norm=norm, weight=weight, bias=bias,
                                 activation=activation, allow_zero_in_degree=allow_zero_in_degree)

    def forward(self, graph, feat, edge_weights):
        with graph.local_scope():
            feat_src, feat_dst = dgl.utils.expand_as_pair(feat, graph)
            weight = self.gcnconv.weight

            if self.gcnconv._in_feats > self.gcnconv._out_feats:
                if weight is not None:
                    if self.feat_drop > 0:
                        feat_src = F.dropout(feat_src, p=self.feat_drop, training=self.training)
                    feat_src = torch.matmul(feat_src, weight)

                graph.srcdata.update({'ft': feat_src})
                graph.edata.update({'a': F.dropout(
                    edge_weights, p=self.aggr_drop, training=self.training)})
                # graph.edata.update({'a': edge_weights * edge_dropout(
                #     graph, edge_weights, p=self.aggr_drop, training=self.training)})
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'h'))
                rst = graph.dstdata['h']
            else:
                graph.srcdata.update({'ft': feat_src})
                graph.edata.update({'a': F.dropout(
                    edge_weights, p=self.aggr_drop, training=self.training)})
                # graph.edata.update({'a': edge_weights * edge_dropout(
                    # graph, edge_weights, p=self.aggr_drop, training=self.training)})
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'h'))
                rst = graph.dstdata['h']

                if weight is not None:
                    if feat_drop > 0:
                        rst = F.dropout(rst, p=self.feat_drop, training=self.training)
                    rst = torch.matmul(rst, weight)

            if self.gcnconv.bias is not None:
                rst = rst + self.gcnconv.bias

            if self.gcnconv._activation is not None:
                rst = self.gcnconv._activation(rst)

            return rst



class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, use_gat=True, attn_dropout=0., adj_norm='right'):
        super(HANLayer, self).__init__()
        self.use_gat = use_gat
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.adj_norm = adj_norm
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            if self.use_gat:
                self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, attn_dropout,
                                               activation=F.elu, allow_zero_in_degree=True))
            else:
                self.gat_layers.append(MeanAggregator(in_size, out_size * layer_num_heads, dropout, attn_dropout, norm='right',
                                                      activation=F.elu, allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self._cached_edge_weight = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            if echo_time:
                torch.cuda.synchronize()
                tic = datetime.datetime.now()
            with torch.no_grad():
                for meta_path in self.meta_paths:
                    temp_graph = dgl.metapath_reachable_graph(g, meta_path)
                    self._cached_coalesced_graph[meta_path] = temp_graph

                    if not self.use_gat:
                        if self.adj_norm == 'none':
                            edge_weights = torch.ones(temp_graph.num_edges())
                        else:
                            src, dst, eid = temp_graph.cpu()._graph.edges(0)
                            adj = SparseTensor(row=dst, col=src, value=eid)
                            # perm = torch.argsort(adj.storage.value())

                            base_ew = 1 / adj.storage.rowcount()
                            base_ew[torch.isnan(base_ew) | torch.isinf(base_ew)] = 0
                            if self.adj_norm == 'right':
                                edge_weights = base_ew[adj.storage.row()]
                            elif self.adj_norm == 'both':
                                edge_weights = torch.pow(base_ew, 0.5)[adj.storage.row()]
                            else:
                                assert False, f'Unknown adj_norm method {self.adj_norm}'

                            # edge_weights = edge_weights[perm]
                        self._cached_edge_weight[meta_path] = edge_weights.unsqueeze(-1).to(device=temp_graph.device)

            if echo_time:
                torch.cuda.synchronize()
                toc = datetime.datetime.now()
                print('Time used for make graph', toc-tic, self.meta_paths, self._cached_coalesced_graph)

        # if echo_time:
        #     torch.cuda.synchronize()
        #     tic = datetime.datetime.now()
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            if self.use_gat:
                semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
            else:
                new_ew = self._cached_edge_weight[meta_path]
                semantic_embeddings.append(self.gat_layers[i](new_g, h, new_ew).flatten(1))
        # if echo_time:
        #     torch.cuda.synchronize()
        #     toc = datetime.datetime.now()
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        out = self.semantic_attention(semantic_embeddings)
        # if echo_time:
        #     torch.cuda.synchronize()
        #     toc2 = datetime.datetime.now()
        #     print(toc2-toc, toc-tic)
        return out


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout, use_gat=True, attn_dropout=0.):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(
            meta_paths, in_size, hidden_size, num_heads[0], dropout, use_gat=use_gat, attn_dropout=attn_dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(
                meta_paths, hidden_size * num_heads[l-1],
                hidden_size, num_heads[l], dropout, use_gat=use_gat, attn_dropout=attn_dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class HAN_freebase(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout, use_gat=True):
        super(HAN_freebase, self).__init__()

        self.layers = nn.ModuleList()
        self.fc = nn.Linear(in_size, hidden_size)
        self.layers.append(HANLayer(meta_paths, hidden_size, hidden_size, num_heads[0], dropout, use_gat=use_gat))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, use_gat=use_gat))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        h = self.fc(h)
        h = torch.FloatTensor(h.cpu()).to(g.device)
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)
