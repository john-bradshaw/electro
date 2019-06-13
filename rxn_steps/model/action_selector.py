
import typing

import torch
from torch import nn

from graph_neural_networks.core import mlp

from ..data import graph_ds


class ActionSelectorInputs(typing.NamedTuple):
    graphs: graph_ds.TorchGraph
    prev_action_per_graph: typing.Optional[torch.Tensor]
    context_vectors_per_graph: typing.Optional[torch.Tensor]


class ActionSelector(nn.Module):
    def __init__(self, mlp_input_size, hidden_sizes):
        super().__init__()
        self.mlp = mlp.MLP(mlp.MlpParams(mlp_input_size, 1, hidden_sizes))

    def forward(self, input_: ActionSelectorInputs) -> typing.Optional[graph_ds.LogitNodeGraph]:
        if input_.graphs is None:
            return None

        stacked_nodes = input_.graphs.node_features  # [v*, h]

        # Create the last action concatenation if applicable  -- done before any other context added to features
        if input_.prev_action_per_graph is not None:
            prev_actions = input_.graphs.graph_offsets + input_.prev_action_per_graph
            prev_action_features = stacked_nodes[prev_actions]
            concat_mat = prev_action_features[input_.graphs.node_to_graphid, :]
            stacked_nodes = torch.cat([stacked_nodes, concat_mat], dim=1)  # [v*, h']

        # Create the graph context concatenation with a predetermined context vector for each graph if applicable
        if input_.context_vectors_per_graph is not None:
            concat_mat = input_.context_vectors_per_graph[input_.graphs.node_to_graphid, :]
            stacked_nodes = torch.cat([stacked_nodes, concat_mat], dim=1)  # [v*, h']

        # Run through the network
        stacked_logits = self.mlp(stacked_nodes)  # [v*, h] -> [v*, 1]
        return graph_ds.LogitNodeGraph(stacked_logits, input_.graphs.node_to_graphid)
