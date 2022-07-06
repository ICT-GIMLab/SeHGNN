import numpy as np
import scipy as sp

import dgl
import dgl.backend as F
from dgl.base import NID, EID

from dglke.dataloader import SoftRelationPartition, RandomPartition


def ConstructGraph(edges, n_entities, args):
    """Construct Graph for training

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list
    n_entities : int
        number of entities
    args :
        Global configs.
    """
    src, etype_id, dst = edges
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
    g = dgl.DGLGraphStale(coo, readonly=True, multigraph=False, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)
    return g


class TrainDataset(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """
    def __init__(self, dataset, args, ranks=64):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)

        if ranks > 1 and args.rel_part:
            self.edge_parts, self.rel_parts, self.cross_part, self.cross_rels = \
            SoftRelationPartition(triples, ranks)
        elif ranks > 1:
            self.edge_parts = RandomPartition(triples, ranks)
            self.cross_part = True
        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = ConstructGraph(triples, dataset.n_entities, args)

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        assert batch_size % neg_sample_size == 0, 'batch_size should be divisible by B'
        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank]),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)


class ChunkNegEdgeSubgraph(dgl.DGLGraphStale):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid


def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)


class NewBidirectionalOneShotIterator:
    """Grouped samper iterator

    Parameters
    ----------
    dataloader_head : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in head mode
    dataloader_tail : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in tail mode
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    neg_sample_size : int
        How many negative edges sampled for each node.
    is_chunked : bool
        If True, the sampled batch is chunked.
    num_nodes : int
        Total number of nodes in the whole graph.
    """
    def __init__(self, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_g, neg_g = next(self.iterator_head)
        else:
            pos_g, neg_g = next(self.iterator_tail)
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes):
        while True:
            for pos_g, neg_g in dataloader:
                neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                            is_chunked, neg_head, num_nodes)
                if neg_g is None:
                    continue

                pos_g.ndata['id'] = pos_g.parent_nid
                neg_g.ndata['id'] = neg_g.parent_nid
                pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                yield pos_g, neg_g
