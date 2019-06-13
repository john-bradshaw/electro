
import typing
import copy
import numpy as np

import torch
from torch.nn import functional as F

from graph_neural_networks.core import utils
from graph_neural_networks.sparse_pattern import graph_as_adj_list

from ..data.rdkit_ops import rdkit_featurization_ops
from ..data.rdkit_ops import rdkit_general_ops
from ..data import graph_ds
from ..model import electro_model
from ..model import get_electro
from ..model import action_selector
from ..misc import data_types

from . import action_sequence


class PredictiveRanking(object):
    def __init__(self, electro: get_electro.FullModel, cuda_details: utils.CudaDetails,
                 num_to_take_forward=10, max_num_steps_out=7):

        self.full_model = electro
        self.electro_lite_flag = self.full_model.electro.reagents_net_aggregator is None
        self.cuda_details = cuda_details
        self.num_to_take_forward = num_to_take_forward
        self.max_num_steps_out = max_num_steps_out
        self.atom_feat_params = rdkit_featurization_ops.AtomFeatParams()

        self._all_stopped_so_far = None

    @torch.no_grad()
    def predict_out(self, input_reactants_atom_mapped_str, reagents_atom_mapped_str):
        self._clear()

        # Set up initial variables
        reagent_context = self._get_reagents_context(reagents_atom_mapped_str)
        initial_molecule = rdkit_general_ops.get_molecule(input_reactants_atom_mapped_str, kekulize=True)
        initial_action_sequence = action_sequence.ActionSequence(initial_molecule,
                                                                 action_sequence.MaskHolders(self.cuda_details.device_str),
                                                                 self.atom_feat_params)
        pool = [initial_action_sequence]
        for t in range(self.max_num_steps_out):

            # 1. Create greaphs and masks for all pool members
            graphs, stop_masks, action_masks = self._get_graphs_and_masks(t, pool)
            graphs_w_node_embeddings = self.full_model.ggnn(graphs)

            # 2. Assess whether to stop for all remaining items in pool
            stop_lprobs = self._get_stop_log_probs_as_np(graphs_w_node_embeddings, stop_masks)

            #   2b.  Add to the stop list all of the action sequences as if they were going to finish at this point
            for action_seq_, log_prob in zip(pool, stop_lprobs):
                action_seq_ = copy.copy(action_seq_)
                action_seq_.log_probs.append(log_prob)
                action_seq_.mark_finished()
                self._all_stopped_so_far.append(action_seq_)
            #   2c. Add the continue prob to all action sequences
            continue_log_probs = np.log(1. - np.exp(stop_lprobs))
            for action_seq_, log_prob in zip(pool, continue_log_probs):
                action_seq_.log_probs.append(log_prob)

            # 3. Assess all actions on the ones in the pool
            pool = self._new_pool_by_considering_all_possible_next_actions(graphs_w_node_embeddings, t, action_masks, pool, reagent_context)

            # 4. Filter the pool so that we keep only the top K (ie the beam width) -- break early if none left
            pool = sorted(pool, reverse=True)[:self.num_to_take_forward]
            if len(pool) == 0:
                break

        return sorted(self._all_stopped_so_far, reverse=True)

    @staticmethod
    def convert_predictions_to_ordered_am_path(list_of_action_seqs: typing.List[action_sequence.ActionSequence]):
        return [elem.actions_as_am for elem in list_of_action_seqs]

    def _new_pool_by_considering_all_possible_next_actions(self, graphs: graph_ds.TorchGraph, t, action_masks, pool, reagent_context):
        # 1. We are going to create the logits, this depends on what step we are on
        if t == 0:  # initial step
            act_select = self.full_model.electro.initial_select
            if reagent_context is not None:
                reagent_context = reagent_context.repeat(graphs.max_num_graphs, 1)
            input_ = action_selector.ActionSelectorInputs(graphs, None, reagent_context)
        elif t % 2 == 1:  # remove step
            act_select = self.full_model.electro.remove_select
            previous_action_indx = self._get_previous_action_indices(pool)
            input_ = action_selector.ActionSelectorInputs(graphs, previous_action_indx, None)
        else:  # must be an add step
            act_select = self.full_model.electro.add_select
            previous_action_indx = self._get_previous_action_indices(pool)
            input_ = action_selector.ActionSelectorInputs(graphs, previous_action_indx, None)

        logits: graph_ds.LogitNodeGraph = act_select(input_)
        padded_log_probs = logits.pad_log_prob_matrix(action_masks).cpu().numpy()

        # 2. We now will go through and create action sequences with all plausible actions
        new_pool = []
        for act_seq_from_old_pool, log_probs_for_actions in zip(pool, padded_log_probs):
            for idx, l_prob in enumerate(log_probs_for_actions):
                if l_prob == -np.inf:
                    continue  # skip as impossible

                new_act_seq = copy.copy(act_seq_from_old_pool)
                new_act_seq.add_an_action(idx, l_prob)
                new_pool.append(new_act_seq)

        return new_pool

    def _get_previous_action_indices(self, pool: typing.List[action_sequence.ActionSequence]):
        action_indcs = [elem.prev_act_idx for elem in pool]
        action_indcs = torch.tensor(action_indcs, dtype=data_types.TORCH_INT)
        action_indcs = self.cuda_details.return_cudafied(action_indcs)
        return action_indcs

    def _get_graphs_and_masks(self, t, pool: typing.List[action_sequence.ActionSequence]):
        graphs, stop_masks, action_masks = zip(*[elem.get_graph_and_masks(t) for elem in pool])

        graphs: typing.List[graph_as_adj_list.GraphAsAdjList] = list(graphs)
        graphs = graphs[0].concatenate(graphs)
        graphs = graphs.to_torch(self.cuda_details)

        stop_masks = np.concatenate(list(stop_masks))
        stop_masks = self.cuda_details.return_cudafied(torch.from_numpy(stop_masks.astype(np.uint8)))

        action_masks = np.stack(action_masks)
        action_masks = self.cuda_details.return_cudafied(torch.from_numpy(action_masks.astype(np.uint8)))
        return graphs, stop_masks, action_masks

    def _get_stop_log_probs_as_np(self, graphs, mask):
        stop_logits = torch.squeeze(self.full_model.electro.stop_net_aggregator(graphs), dim=1)
        stop_log_prob = F.logsigmoid(stop_logits)
        stop_log_prob[~mask] = -np.inf

        stop_log_prob = stop_log_prob.cpu().numpy()
        return stop_log_prob

    def _clear(self):
        self._all_stopped_so_far = []

    def _get_reagents_context(self, reagents_atom_mapped_str):
        if self.electro_lite_flag:
            reagent_context = None
        else:
            reagent_mol = rdkit_general_ops.get_molecule(reagents_atom_mapped_str, kekulize=True)
            if rdkit_general_ops.get_num_atoms(reagent_mol) == 0:
                reagent_graph = rdkit_featurization_ops.create_empty_graph(self.atom_feat_params)
            else:
                reagent_graph = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(
                    reagent_mol,
                    params=self.atom_feat_params)
            reagent_graph = reagent_graph.to_torch(self.cuda_details)
            reagent_graph = self.full_model.ggnn(reagent_graph)
            reagent_context = self.full_model.electro.reagents_net_aggregator(reagent_graph)
            assert reagent_context.shape[0] == 1, "More than one reagent!"
        return reagent_context
