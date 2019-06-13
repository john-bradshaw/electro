
import typing

import torch
from torch import nn


from ..data import graph_ds
from . import action_selector
from . import graph_models



class Electro(nn.Module):
    def __init__(self,
                 stop_net_aggregator: graph_models.GraphAggregator,
                 reagents_net_aggregator: typing.Optional[graph_models.GraphAggregator],
                 initial_select: action_selector.ActionSelector,
                 remove_select: action_selector.ActionSelector,
                 add_select: action_selector.ActionSelector):
        super().__init__()
        self.stop_net_aggregator = stop_net_aggregator
        self.reagents_net_aggregator = reagents_net_aggregator
        self.initial_select = initial_select
        self.remove_select = remove_select
        self.add_select = add_select

    def forward(self, graphs: graph_ds.TorchGraph,
                initial_select_inputs: graph_ds.ActionSelectorGraphIds,
                remove_select_inputs: graph_ds.ActionSelectorGraphIds,
                add_select_inputs: graph_ds.ActionSelectorGraphIds,
                reagent_graphs: typing.Optional[graph_ds.TorchGraph]=None):

        # 1. Compute the stop logits
        stop_logits = self.stop_net_aggregator(graphs)
        stop_logits = torch.squeeze(stop_logits, dim=1)

        # 2. Compute the reagent context if necessary
        if self.reagents_net_aggregator is not None:
            assert reagent_graphs is not None
            reagent_context = self.reagents_net_aggregator(reagent_graphs)
        else:
            reagent_context = None

        # 3. Then do the initial, remove, and add actions
        results = []
        for action_selector_input, selector in [(initial_select_inputs, self.initial_select),
                                                       (remove_select_inputs, self.remove_select),
                                                       (add_select_inputs, self.add_select)]:
            # 3a. Pick out the graphs we're interested in
            graphs_of_interest = graphs[action_selector_input.graphs_ids]

            # If there are no graphs of interest then we need not do any more calculations ...
            if graphs_of_interest is None:
                results.append(graph_ds.LogitNodeGraph.create_empty(graphs.dev_str_feats))

            # ... but if there are then we need to do more.
            else:
                # 3b. We select the reagents context vectors for the graphs we are interested in
                reagents_necessary = ((reagent_context is not None) and
                        (action_selector_input.reagent_context_ids is not None) and
                        (action_selector_input.reagent_context_ids.shape[0] != 0))
                if reagents_necessary:
                    context = reagent_context[action_selector_input.reagent_context_ids]
                else:
                    context = None

                # 3c. We create the input data structure by selecting out the applicable graphs
                inp = action_selector.ActionSelectorInputs(graphs_of_interest,
                                                           action_selector_input.prev_action_per_graph, context)
                # 3d. We then compute the logits for the different actions.
                logits = selector(inp)
                results.append(logits)
        initial_action_logits,  remove_action_logits, add_action_logits = results
        return stop_logits, initial_action_logits, remove_action_logits, add_action_logits


