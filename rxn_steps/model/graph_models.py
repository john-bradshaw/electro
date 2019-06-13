

import typing

from torch import nn

from ..data import graph_ds

from graph_neural_networks.ggnn_general import graph_tops
from graph_neural_networks.sparse_pattern import ggnn_sparse
from graph_neural_networks.ggnn_general import ggnn_base
from graph_neural_networks.sparse_pattern import graph_as_adj_list
from graph_neural_networks.core import mlp
from graph_neural_networks.core import utils

from ..data import graph_ds


class NodeEmbedder(nn.Module):
    """
    Takes node features and graph structure and computes node embeddings from this.
    """
    def __init__(self, hidden_layer_size: int, edge_names: typing.List[str], embedding_dim: int,
                 cuda_details: utils.CudaDetails, num_time_steps: int):
        super().__init__()
        self.ggnn = ggnn_sparse.GGNNSparse(
            ggnn_base.GGNNParams(hidden_layer_size, edge_names, cuda_details, num_time_steps))

        self.embedding_dim = embedding_dim

    def forward(self, g_adjlist: graph_as_adj_list.GraphAsAdjList):
        g_adjlist: graph_as_adj_list.GraphAsAdjList = self.ggnn(g_adjlist)
        graph_out = graph_ds.TorchGraph(g_adjlist.node_features, node_to_graphid=g_adjlist.node_to_graph_id)
        return graph_out


class GraphAggregator(nn.Module):
    """
    Takes node embeddings and details of which graph they belong to and aggregates them through an attention weighted
    sum to form a graph level embedding.

    Mostly a convenience wrapper on graph_tops.GraphFeaturesStackIndexAdd
    """
    def __init__(self, node_feature_dim: int, final_dim: int, cuda_details: utils.CudaDetails):
        super().__init__()
        mlp_up = mlp.MLP(mlp.MlpParams(node_feature_dim, 2*node_feature_dim, []))
        mlp_gate = mlp.MLP(mlp.MlpParams(node_feature_dim, 1, []))
        mlp_func = mlp.MLP(mlp.MlpParams(2*node_feature_dim, final_dim, []))

        self.g_top = graph_tops.GraphFeaturesStackIndexAdd(mlp_up, mlp_gate, mlp_func, cuda_details=cuda_details)

    def forward(self, graphs: graph_ds.TorchGraph):
        results = self.g_top(graphs.node_features, graphs.node_to_graphid)
        return results
