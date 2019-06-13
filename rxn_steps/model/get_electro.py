
import enum
import typing

from torch import nn

from graph_neural_networks.core import utils

from ..data.rdkit_ops import rdkit_featurization_ops
from . import electro_model
from . import graph_models
from . import action_selector


class ElectroVariants(enum.Enum):
    ELECTRO = 'ELECTRO'
    ELECTRO_LITE = 'ELECTRO-LITE'




class FullModel(nn.Module):
    def __init__(self, ggnn, electro):
        super().__init__()
        self.ggnn: graph_models.NodeEmbedder =  ggnn
        self.electro: electro_model.Electro = electro


def get_model(variant: ElectroVariants, cuda_details: utils.CudaDetails):
    chem_details = rdkit_featurization_ops.AtomFeatParams()
    node_embedding_size = chem_details.atom_feature_length
    ggnn = graph_models.NodeEmbedder(hidden_layer_size=node_embedding_size,
                                     edge_names=chem_details.bond_names,
                                     embedding_dim=node_embedding_size, cuda_details=cuda_details,
                                     num_time_steps=4)

    if variant is ElectroVariants.ELECTRO:
        reagents_cntxt_size = 100

        initial_select = action_selector.ActionSelector(node_embedding_size + reagents_cntxt_size, [100, 100])
        remove_select = action_selector.ActionSelector(2*node_embedding_size, [100])
        add_select = action_selector.ActionSelector(2*node_embedding_size, [100])
        stop_net = graph_models.GraphAggregator(node_embedding_size, 1, cuda_details)

        reagent_context_net = graph_models.GraphAggregator(node_embedding_size,
                                                           reagents_cntxt_size, cuda_details)
    elif variant is ElectroVariants.ELECTRO_LITE:
        initial_select = action_selector.ActionSelector(node_embedding_size, [100])
        remove_select = action_selector.ActionSelector(2*node_embedding_size, [100])
        add_select = action_selector.ActionSelector(2*node_embedding_size, [100])
        stop_net = graph_models.GraphAggregator(node_embedding_size, 1, cuda_details)
        reagent_context_net = None

    else:
        raise NotImplementedError
    electro = electro_model.Electro(stop_net, reagent_context_net, initial_select, remove_select, add_select)
    return FullModel(ggnn, electro)

