
import typing

from dataclasses import dataclass
import torch
import numpy as np
from torch_scatter import scatter_max

from graph_neural_networks.core import nd_ten_ops

from ..misc import data_types
from ..misc import misc_torch_funcs


class TorchGraph:
    """
    This stores node features belonging to all graphs stacked one on top of each other (so we do not need to pad, to
    deal with graphs with different number of nodes).

    Nodes belonging to each graph should be all grouped together.
    With each node we also store its graph id. Ie what graph it belongs to.
    The graph ids should be in order, consecutive integers and start from 0 (ie similar to normal indexing)
    """
    def __init__(self, node_features: torch.tensor, node_to_graphid: torch.tensor,
                 num_nodes_per_graph: typing.Optional[torch.Tensor]=None):
        # Note that all nodes for each graph should be next to each other -- but this will not be checked.
        self.node_features = node_features  # [v*, h]
        self.node_to_graphid = node_to_graphid  # [v*]

        self.empty_graph = self.node_features.shape[0] == 0

        if not self.empty_graph:
            if num_nodes_per_graph is None:
                self.num_nodes_per_graph = self._get_the_number_of_nodes_per_graph()
            else:
                self.num_nodes_per_graph = num_nodes_per_graph
            self.graph_offsets = torch.cat([torch.tensor([0], dtype=data_types.TORCH_INT, device=self.dev_str_n2gid),
                                            torch.cumsum(self.num_nodes_per_graph[:-1], 0)])

    @property
    def max_num_graphs(self):
        return self.node_to_graphid.max() + 1  # plus one as we index from 0.

    @property
    def dev_str_n2gid(self):
        return str(self.node_to_graphid.device)

    @property
    def dev_str_feats(self):
        return str(self.node_features.device)

    def __getitem__(self, graph_ids_of_interest):
        """
        Get a TorchGraph which represents nodes only in those graph_ids_of_interest. It also reindexes the graph ids
        so that we only use a consecutive set starting from zero.

        If the indexing tensor is empty we return None.
        """
        no_graphs_selected = graph_ids_of_interest.shape[0] == 0
        if self.empty_graph or no_graphs_selected:
            return None
        else:
            # Convert graph id to their locations in the stacked list
            nodes_to_use_mask = misc_torch_funcs.isin(self.node_to_graphid, graph_ids_of_interest)

            # Now index out the parts corresponding to the graphs we're interested in:
            new_node_features = self.node_features[nodes_to_use_mask, :]
            num_nodes_per_graph = self.num_nodes_per_graph[graph_ids_of_interest]

            # Now we create readjust the graph ids so that they are consecutive beginning with zeros
            new_graph_ids = torch.zeros(self.max_num_graphs, device=self.dev_str_n2gid, dtype=data_types.TORCH_INT)
            new_graph_ids[graph_ids_of_interest] = torch.arange(0, graph_ids_of_interest.shape[0], device=self.dev_str_n2gid,
                                                                dtype=data_types.TORCH_INT)
            old_node_to_graph_ids_of_those_using = self.node_to_graphid[nodes_to_use_mask]
            new_node_to_graph_id = new_graph_ids[old_node_to_graph_ids_of_those_using]

            # Now create the new datastructure
            grph = TorchGraph(new_node_features, new_node_to_graph_id, num_nodes_per_graph)
            return grph

    def __setitem__(self, key, value):
        raise NotImplementedError

    def _get_the_number_of_nodes_per_graph(self):
        dev_str = self.dev_str_n2gid
        ones = torch.ones(self.node_features.shape[0], device=dev_str, dtype=data_types.TORCH_INT)
        graph_sums = torch.zeros(self.max_num_graphs, device=dev_str, dtype=data_types.TORCH_INT)
        graph_sums.index_add_(0, self.node_to_graphid, ones)
        return graph_sums


class LogitNodeGraph(TorchGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.empty_graph:
            assert self.node_features.shape[1] == 1, "The node features should now be logits"

    @classmethod
    def create_empty(cls, device_str):
        return cls(torch.tensor([], device=device_str), torch.tensor([], device=device_str))

    @property
    def squeezed_logits(self):
        return torch.squeeze(self.node_features, dim=1)

    def nll_per_graph(self, true_action_per_graph: torch.Tensor, stacked_mask: torch.Tensor):
        if self.empty_graph:
            return torch.tensor([], dtype=data_types.TORCH_FLT, device=self.dev_str_feats)
        assert true_action_per_graph.shape[0] == self.max_num_graphs

        # Take off the maximum to try to avoid overflow after exp, see
        # eg https://stackoverflow.com/questions/42599498/numercially-stable-softmax
        max_logits, _ = scatter_max(src=self.squeezed_logits, index=self.node_to_graphid, fill_value=self.node_features.min().item())
        max_logits = max_logits.detach()

        constant_shifted_logits = self.squeezed_logits - max_logits[self.node_to_graphid]
        constant_shifted_logits[~stacked_mask] = -np.inf # these have zero probability by mask def

        # Calculate the softmax -- work out the exponential termms
        exp_shift_logits = torch.exp(constant_shifted_logits)

        # Calculate the denominator
        num_graphs = self.max_num_graphs
        denom = torch.zeros(num_graphs, dtype=self.node_features.dtype, device=self.dev_str_feats)
        denom.index_add_(dim=0, index=self.node_to_graphid, source=exp_shift_logits)

        # Correct class exponential logit
        labels_of_correct = true_action_per_graph + self.graph_offsets
        nll = torch.log(denom) - constant_shifted_logits[labels_of_correct]

        return nll

    def pad_log_prob_matrix(self, stacked_mask: torch.Tensor):
        if self.empty_graph:
            raise RuntimeError("Cannot produce probabilities when no logits.")

        # Create a padded array ready to turn as well as a place to store the invalid points
        height = self.max_num_graphs
        width = self.num_nodes_per_graph.max()
        op = torch.zeros(height, width, device=self.dev_str_feats, dtype=self.node_features.dtype)
        valid_mask = torch.zeros_like(op, dtype=torch.uint8)

        # now go though and populate each row of these padded matrices with their values
        squeezed_logits = self.squeezed_logits
        for i in range(self.max_num_graphs):
            locs = self.node_to_graphid == i
            vals = squeezed_logits[locs]
            mask = stacked_mask[i]
            num_graphs = self.num_nodes_per_graph[i]
            assert vals.shape[0] == num_graphs, "the logits should match number of nodes in graphs"
            assert mask.shape[0] == width.item(), \
                "the stacked mask for this graph should match the maximum number of nodes in a graph in this batch"
            op[i, :num_graphs] = vals
            valid_mask[i, :] = mask

        # Before doing the softmax we take off the maximum value (nb how value of softmax is not affected by adding a
        # constant to the logits). This hopefully improves stability by avoiding very large exponentials.
        #todo: make sure that we are not getting a stability hit at the bottom end from selecting as the max invalid logits
        max_logit = torch.max(op.detach(), dim=1, keepdim=True)[0]
        logit = op - max_logit
        exp_logit = torch.exp(logit)

        exp_logit = exp_logit * valid_mask.type(exp_logit.dtype)
        log_probs = torch.log(exp_logit) - torch.log(torch.sum(exp_logit, dim=1, keepdim=True))
        return log_probs


@dataclass
class ActionSelectorGraphIds:
    graphs_ids: nd_ten_ops.Nd_Ten
    prev_action_per_graph: nd_ten_ops.Op_Nd_Ten
    reagent_context_ids: nd_ten_ops.Op_Nd_Ten


@dataclass
class ActionSelectorTarget:
    label_per_graph: nd_ten_ops.Nd_Ten
    action_mask: nd_ten_ops.Nd_Ten
