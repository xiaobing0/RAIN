"""Data loading components for neighbor sampling"""
from .dataloader import BlockSampler
from .. import sampling, subgraph, distributed
from .. import ndarray as nd
from .. import backend as F
from ..base import ETYPE
import dgl
import torch
import numpy as np

class MultiLayerNeighborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int] or None]
        List of neighbors to sample per edge type for each GNN layer, starting from the
        first layer.

        If the graph is homogeneous, only an integer is needed for each layer.

        If None is provided for one layer, all neighbors will be included regardless of
        edge types.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    replace : bool, default True
        Whether to sample with replacement
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the MFG.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 15])
    >>> collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """
    def __init__(self, fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace

        # used to cache computations and memory allocations
        # list[dgl.nd.NDArray]; each array stores the fan-outs of all edge types
        self.fanout_arrays = []
        self.prob_arrays = None

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if isinstance(g, distributed.DistGraph):
            if fanout is None:
                # TODO(zhengda) There is a bug in the distributed version of in_subgraph.
                # let's use sample_neighbors to replace in_subgraph for now.
                frontier = distributed.sample_neighbors(g, seed_nodes, -1, replace=False)
            else:
                if len(g.etypes) > 1: # heterogeneous distributed graph
                    # The edge type is stored in g.edata[dgl.ETYPE]
                    assert isinstance(fanout, int), "For distributed training, " \
                        "we can only sample same number of neighbors for each edge type"
                    frontier = distributed.sample_etype_neighbors(
                        g, seed_nodes, ETYPE, fanout, replace=self.replace)
                else:
                    frontier = distributed.sample_neighbors(
                        g, seed_nodes, fanout, replace=self.replace)
        else:
            if fanout is None:
                frontier = subgraph.in_subgraph(g, seed_nodes)
            else:
                self._build_fanout(block_id, g)
                self._build_prob_arrays(g)

                frontier = sampling.sample_neighbors(
                    g, seed_nodes, self.fanout_arrays[block_id],
                    replace=self.replace, prob=self.prob_arrays)
        return frontier

    def _build_prob_arrays(self, g):
        # build prob_arrays only once
        if self.prob_arrays is None:
            self.prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)

    def _build_fanout(self, block_id, g):
        assert not self.fanouts is None, \
            "_build_fanout() should only be called when fanouts is not None"
        # build fanout_arrays only once for each layer
        while block_id >= len(self.fanout_arrays):
            for i in range(len(self.fanouts)):
                fanout = self.fanouts[i]
                if not isinstance(fanout, dict):
                    fanout_array = [int(fanout)] * len(g.etypes)
                else:
                    if len(fanout) != len(g.etypes):
                        raise DGLError('Fan-out must be specified for each edge type '
                                       'if a dict is provided.')
                    fanout_array = [None] * len(g.etypes)
                    for etype, value in fanout.items():
                        fanout_array[g.get_etype_id(etype)] = value
                self.fanout_arrays.append(
                    F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64)))


class MultiLayerFullNeighborSampler(MultiLayerNeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    n_layers : int
        The number of GNN layers to sample.
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the MFG.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """
    def __init__(self, n_layers, return_eids=False):
        super().__init__([None] * n_layers, return_eids=return_eids)

class MultiLayerDropoutSampler(BlockSampler):
    def __init__(self, num_layers):
        super().__init__(num_layers)
        # self.fea = fea
        # self.p = p

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # 获取种 `seed_nodes` 的所有入边
        src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
        # if len(src) < 10000:  # reddit
        #     frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        # # else:
        # #     mask = torch.zeros_like(src).bernoulli_(self.p).bool()
        # #     src = src[mask]
        # #     dst = dst[mask]
        # #     frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        # # else:
        # #     mask = torch.zeros_like(src).bool()
        # #     # 每个边一个权重,从所有的边里按权重的概率取原来边的数目的p倍,p小于1
        # #     # mask_m = list(torch.utils.data.WeightedRandomSampler([np.linalg.norm(self.fea[src[ind_e]]- self.fea[dst[ind_e]],
        # #     #                  ord=1, axis=None, keepdims=False)+1 for ind_e in range(len(src))], 20000, replacement=False))
        # #     mask_m = list(
        # #         torch.utils.data.WeightedRandomSampler([1/g.in_degrees(dst[ind_e]) for ind_e in range(len(src))], 20000, replacement=False))
        # #     mask[mask_m] = 1
        # #     src = src[mask]
        # #     dst = dst[mask]
        # #     frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        #
        # elif 10000 < len(src) < 50000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 35)
        # elif 50000 < len(src) < 100000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 40)
        # elif 100000 < len(src) < 150000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 55)
        # else:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 80)
        # return frontier

        # if len(src) < 10000:  # product
        #     frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        # elif 10000 < len(src) < 30000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 20)
        # elif 30000 < len(src) < 60000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 25)
        # elif 60000 < len(src) < 100000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 30)
        # else:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 40)
        # return frontier

        # if len(src) < 10000:  # yel
        #     frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        # elif 10000 < len(src) < 20000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 25)
        # elif 20000 < len(src) < 60000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 30)
        # elif 60000 < len(src) < 100000:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 50)
        # else:
        #     frontier = sampling.sample_neighbors(g, seed_nodes, 70)
        # return frontier

        if len(src) < 10000:  # amazon
            frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        elif 10000 < len(src) < 20000:
            frontier = sampling.sample_neighbors(g, seed_nodes, 25)
        elif 20000 < len(src) < 60000:
            frontier = sampling.sample_neighbors(g, seed_nodes, 30)
        elif 60000 < len(src) < 100000:
            frontier = sampling.sample_neighbors(g, seed_nodes, 50)
        else:
            frontier = sampling.sample_neighbors(g, seed_nodes, 70)
        return frontier



    def __len__(self):
        return self.num_layers

class MultiLayerDropoutSampler_2(BlockSampler):
    def __init__(self, p, num_layers):
        super().__init__(num_layers)
        self.p = p

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # 获取种 `seed_nodes` 的所有入边
        src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
        # mask = torch.zeros_like(src).bernoulli_(0.8).bool()
        # src = src[mask]
        # dst = dst[mask]
        # frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())

        if len(src) < 120000:
            # frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
            # frontier = sampling.sample_neighbors(g, seed_nodes, 10)
            mask = torch.zeros_like(src).bernoulli_(0.96).bool()
            src = src[mask]
            dst = dst[mask]
            frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        else:
            mask = torch.zeros_like(src).bernoulli_(0.9).bool()
            src = src[mask]
            dst = dst[mask]
            frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
            frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        return frontier



    def __len__(self):
        return self.num_layers