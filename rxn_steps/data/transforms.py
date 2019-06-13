
import typing
import itertools

import numpy as np
import torch
from rdkit import Chem

from graph_neural_networks.sparse_pattern import graph_as_adj_list
from graph_neural_networks.core import nd_ten_ops
from graph_neural_networks.core import utils

from .rdkit_ops import rdkit_reaction_ops
from .rdkit_ops import rdkit_general_ops
from .rdkit_ops import rdkit_featurization_ops
from .rdkit_ops import chem_details
from . import lef_consistency_errors
from . import action_list_funcs
from . import mask_creators
from . import graph_ds
from ..misc import data_types


"""
 ==== Datastructures from the different transforms ====
"""


class BrokenDownParts(typing.NamedTuple):
    reactants: str  # AM-SMILES
    reagents: str  # AM-SMILES
    products: str  # AM-SMILES
    ordered_am_path: typing.List[int]  # should include self-bond break at start if applicable by repeated action.


class RdKitIntermediates(typing.NamedTuple):
    intermediates: typing.List[Chem.Mol]
    reagents: Chem.Mol
    ordered_indx_path: typing.List[int]
    am_to_idx_map: typing.Mapping[int, int]


class ElectroTrainInput:
    def __init__(self,
            graphs: graph_as_adj_list.GraphAsAdjList,
            reagents: graph_as_adj_list.GraphAsAdjList,

            initial_input: graph_ds.ActionSelectorGraphIds,
            remove_input: graph_ds.ActionSelectorGraphIds,
            add_input: graph_ds.ActionSelectorGraphIds,

            initial_target: graph_ds.ActionSelectorTarget,
            remove_target: graph_ds.ActionSelectorTarget,
            add_target: graph_ds.ActionSelectorTarget,

            stop_label: nd_ten_ops.Nd_Ten,
            stop_mask: nd_ten_ops.Nd_Ten,

            num_reactions:int):
        self.graphs = graphs
        self.reagents = reagents

        self.initial_input = initial_input
        self.remove_input = remove_input
        self.add_input = add_input

        self.initial_target = initial_target
        self.remove_target = remove_target
        self.add_target = add_target

        self.stop_label = stop_label
        self.stop_mask = stop_mask
        self.num_reactions = num_reactions

    def to_torch(self, cuda_details: utils.CudaDetails):
        self.graphs = self.graphs.to_torch(cuda_details)
        self.reagents = self.reagents.to_torch(cuda_details)
        def func(op_arr: typing.Optional[np.ndarray]):
            if op_arr is not None:
                if op_arr.dtype == np.bool_:
                    op_arr = op_arr.astype(np.uint8)  # torch has no bool_ type.
                return torch.from_numpy(op_arr).to(cuda_details.device_str)
            else:
                return None

        self._over_nd_ten_fields(func)
        return self

    def _over_nd_ten_fields(self, func):
        for inp_ in [self.initial_input, self.remove_input, self.add_input]:
            inp_.graphs_ids = func(inp_.graphs_ids)
            inp_.reagent_context_ids = func(inp_.reagent_context_ids)
            inp_.prev_action_per_graph = func(inp_.prev_action_per_graph)

        for inp_ in [self.initial_target, self.remove_target, self.add_target]:
            inp_.action_mask = func(inp_.action_mask)
            inp_.label_per_graph = func(inp_.label_per_graph)

        self.stop_label = func(self.stop_label)
        self.stop_mask = func(self.stop_mask)



"""
 ==== Transforms between the different datastructures ====
"""

class TransformStrToBrokenDownParts:
    """
    (A) Creates path and assigns order see Fig 3 of the paper.
    (B) Breaks out the reagents,

    It runs the following consistency checks:
    C1. The atoms can be lined up end to end on the path
    """
    def __call__(self, input_: typing.Tuple[str, str, str]) -> BrokenDownParts:
        reactants, products, bond_changes = input_

        # A.2 We first create the path (checks for C1.)
        change_list = bond_changes.split(';')
        atom_pairs = np.array([c.split('-') for c in change_list]).astype(int)
        unordered_electron_path_am = action_list_funcs.actions_am_from_pairs(atom_pairs, consistency_check=True)

        # B Using these actions we can now work out what is a reagents and split these out.
        reactants, reagents, products = rdkit_reaction_ops.split_reagents_out_from_reactants_and_products(
            reactants, products, unordered_electron_path_am)

        # A.3 We can now order the electron path.
        reactant_mol = rdkit_general_ops.get_molecule(reactants, kekulize=False)
        product_mol = rdkit_general_ops.get_molecule(products, kekulize=False)
        ordered_electon_path_am = action_list_funcs.order_actions_am(unordered_electron_path_am, reactant_mol, product_mol)

        # A.3b Work out whether to add add a self-bond remove at the start
        # (we need to start with a remove action to pick up a pair of electrons)
        first_bond_in_reactants = rdkit_general_ops.get_bond_double_between_atom_mapped_atoms(reactant_mol, ordered_electon_path_am[0],
                                                                                     ordered_electon_path_am[1])
        first_bond_in_products = rdkit_general_ops.get_bond_double_between_atom_mapped_atoms(product_mol, ordered_electon_path_am[0],
                                                                                    ordered_electon_path_am[1])
        starts_already_with_remove_bond = first_bond_in_reactants - first_bond_in_products > 0
        if not starts_already_with_remove_bond:
            ordered_electon_path_am = [ordered_electon_path_am[0]] + ordered_electon_path_am

        # We can now create the o/p data-structure
        op = BrokenDownParts(reactants, reagents, products, ordered_electon_path_am)
        return op


class TransformToRdKitIntermediates:
    """
    This creates a series of intermediate molecules showing the molecule state at each stage of the reaction path.
    Effectively this is panel 4 of Fig3 of the the paper.

    It runs the following consistency checks:
    C2. The bond differences between the reactants and products along the path vary by +- one
    C3. The final molecule created by editing the reactants should be consistent with reported product.
         However, it can contain more information, eg minor products.
        we therefore check consistency of "super molecule" formed by changing actions versus the "sub molecule",
         which is the product given in the dataset. The super molecule should have disconnected extra molecules but
         the major product should show up in it.
    """
    def __call__(self, reaction_parts: BrokenDownParts) -> RdKitIntermediates:
        # 1. Work out the number of add/remove steps.
        # We always start on a remove bond (can be remove self bond though) and go in intermediate add/remove steps)
        # Therefore the add remove steps is as follows:
        action_types = [bond_change for bond_change, _ in
                 zip(
                     itertools.cycle([chem_details.ElectronMode.REMOVE, chem_details.ElectronMode.ADD]),
                     range(len(reaction_parts.ordered_am_path) -1)
                 )]

        # 2. Form the intermediate states
        reactant_mol = rdkit_general_ops.get_molecule(reaction_parts.reactants, kekulize=True)
        reactant_atom_map_to_idx_map = rdkit_general_ops.create_atom_map_indcs_map(reactant_mol)
        intermediates = [reactant_mol, reactant_mol]  # twice as after picked up half a pair nothing has happened.
        action_pairs = zip(reaction_parts.ordered_am_path[:-1], reaction_parts.ordered_am_path[1:])
        for step_mode, (start_atom_am, next_atom_am) in zip(action_types, action_pairs):
            # Nb note we do not change any Hydrogen numbers on the first and last step -- these may change in practice
            # eg gaining H from water in solution, however, we do not represent Hydrogens in our graph structure,
            # apart from in the features.
            prev_mol = intermediates[-1]
            start_atom_idx = reactant_atom_map_to_idx_map[start_atom_am]
            next_atom_idx = reactant_atom_map_to_idx_map[next_atom_am]
            if start_atom_idx == next_atom_idx:
                # then a self bond removal:
                intermed = rdkit_reaction_ops.change_mol_atom(prev_mol, step_mode, start_atom_idx)
            else:
                intermed = rdkit_reaction_ops.change_mol_bond(prev_mol, step_mode, (start_atom_idx, next_atom_idx))
            intermediates.append(intermed)

        # 3. Form the reagent molecule
        reagent_mol = rdkit_general_ops.get_molecule(reaction_parts.reagents, kekulize=True)

        # 4. Consistency checks.
        product_mol = rdkit_general_ops.get_molecule(reaction_parts.products)
        # we first check C2:
        if not rdkit_reaction_ops.is_it_alternating_add_and_remove_steps(reactant_mol, product_mol,
                                                                         reaction_parts.ordered_am_path):
            raise lef_consistency_errors.NotAddingAndRemovingError

        if not rdkit_reaction_ops.is_sub_mol_consistent_with_super(product_mol, intermediates[-1]):
            raise lef_consistency_errors.InconsistentActionError(f"Inconsistent action error for molecule:"
                                                                 f" {[str(part) for part in reaction_parts]}")

        # 5. Switch ordered path to refer to atoms indices rather than atom mapped number.
        ordered_index_path = [reactant_atom_map_to_idx_map[am] for am in reaction_parts.ordered_am_path]

        # 6. Create the final output
        op = RdKitIntermediates(intermediates, reagent_mol, ordered_index_path, reactant_atom_map_to_idx_map)
        return op


class TransformRdKitIntermediatesToElectroTrainInput:
    """
    Takes the intermediate molecules and produces the inputs that we need for training ELECTRO.

    This consists of:
    1. Converting all intermediates to graphs that are easy to run through a GNN.
    2. Converting the reagents to a separate graph (or setting it as None if we choose to bin them)
    3. Get the graphids of initial, remove and add actions
    4. Compute the masks for stop and the atom actions.
    5. Create the targets for stop and atom actions.
    6. Split these different actions up (putting into datastructures)
    """
    def __init__(self, exclude_reagents_from_initial_flag):
        self.exclude_reagents_from_initial_flag = exclude_reagents_from_initial_flag
        self.atom_feat_params = rdkit_featurization_ops.AtomFeatParams()

        # mask creators
        self.initial_mask = mask_creators.InitialActionMaskCreator('cpu')
        self.add_mask = mask_creators.AddMaskCreator('cpu')
        self.remove_mask = mask_creators.RemoveMaskCreator('cpu')
        self.stop_mask = mask_creators.StopMaskCreator('cpu')

    def __call__(self, reaction_rdkit_mols: RdKitIntermediates) -> ElectroTrainInput:
        # 1. Convert all the intermediates to graphs
        mol_list = [mol for mol in reaction_rdkit_mols.intermediates]
        graphs_uncat = [rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(mol, reaction_rdkit_mols.am_to_idx_map,
                                                                         params=self.atom_feat_params)
            for mol in reaction_rdkit_mols.intermediates]
        graphs = graphs_uncat[0].concatenate(graphs_uncat)

        # 2. Convert reagent to graph
        if rdkit_general_ops.get_num_atoms(reaction_rdkit_mols.reagents) == 0:
            reagent_graph = rdkit_featurization_ops.create_empty_graph(self.atom_feat_params)
        else:
            reagent_graph = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(reaction_rdkit_mols.reagents,
                                                                     params=self.atom_feat_params)
        reagent_graph_id = np.array([0])
        init_reagent_info = None if self.exclude_reagents_from_initial_flag else reagent_graph_id

        # 3.a Get the graphids of initial, remove and add actions
        initial_graphid = np.array([0])
        assert graphs.max_num_graphs == len(reaction_rdkit_mols.ordered_indx_path) + 1
        number_of_graphs_excluding_one_we_should_stop_on = graphs.max_num_graphs-1
        remove_graphids = np.arange(1, number_of_graphs_excluding_one_we_should_stop_on, 2)
        add_graphids = np.arange(2, number_of_graphs_excluding_one_we_should_stop_on, 2)

        # 3.b Get the previous atom for remove and add
        prev_action_idx_remove = np.array([reaction_rdkit_mols.ordered_indx_path[i-1] for i in remove_graphids], dtype=np.int64)
        prev_action_idx_add = np.array([reaction_rdkit_mols.ordered_indx_path[i-1] for i in add_graphids], dtype=np.int64)

        # 4. Compute the masks
        initial_mask = np.concatenate([self.initial_mask(mol_list[i]) for i in initial_graphid])
        remove_mask = np.concatenate([self.remove_mask(mol_list[i], int(reaction_rdkit_mols.ordered_indx_path[i-1]),
                                                   first_step_flag=i==1) for i in remove_graphids])
        add_mask = np.concatenate([self.add_mask(mol_list[i], reaction_rdkit_mols.ordered_indx_path[i - 1])
                              for i in add_graphids]) if len(add_graphids) else np.array([], dtype=np.bool_)
        stop_mask = self.stop_mask.all_time_steps_mask(action_list_ordered=reaction_rdkit_mols.ordered_indx_path)

        # 5. Compute the targets
        initial_target = np.array([reaction_rdkit_mols.ordered_indx_path[0]], dtype=data_types.INT)
        remove_target = np.array([reaction_rdkit_mols.ordered_indx_path[i] for i in remove_graphids], dtype=data_types.INT)
        add_target = np.array([reaction_rdkit_mols.ordered_indx_path[i] for i in add_graphids], dtype=data_types.INT)
        stop_target = np.zeros_like(stop_mask, dtype=data_types.INT)
        stop_target[-1] = 1  # we want to stop at the end!

        # 6. Put into the required datastructures.
        op = ElectroTrainInput(graphs, reagent_graph,
                                   initial_input=graph_ds.ActionSelectorGraphIds(initial_graphid, None, init_reagent_info),
                                   remove_input=graph_ds.ActionSelectorGraphIds(remove_graphids, prev_action_idx_remove, None),
                                   add_input=graph_ds.ActionSelectorGraphIds(add_graphids, prev_action_idx_add, None),
                                   initial_target=graph_ds.ActionSelectorTarget(initial_target, initial_mask),
                                   remove_target=graph_ds.ActionSelectorTarget(remove_target, remove_mask),
                                   add_target=graph_ds.ActionSelectorTarget(add_target, add_mask),
                                   stop_label=stop_target,
                                   stop_mask=stop_mask,
                                   num_reactions=1
                               )
        return op


class ElectroBatchCollate:
    """
    Takes the individual training items spat out by the datasets and collates them together eg collate_fn arg of
     dataloader.
    """
    def __call__(self, inp: typing.Iterable[ElectroTrainInput]) -> ElectroTrainInput:
        # Graphs
        all_graphs = [elem.graphs for elem in inp]
        num_graphs_per_inp = [elem.max_num_graphs for elem in all_graphs]
        graph_offset = [0] + np.cumsum(num_graphs_per_inp[:-1]).tolist()
        new_graphs = all_graphs[0].concatenate(all_graphs)

        # Reagents
        all_reagents = [elem.reagents for elem in inp]
        reagent_sizes = set([elem.max_num_graphs if elem is not None else 0 for elem in all_reagents])
        set_elem = reagent_sizes.pop()
        assert len(reagent_sizes) == 0 and set_elem == 1, "not all graphs have one reagent"
        new_reagents = all_reagents[0].concatenate(all_reagents)

        # New action inputs
        new_initial_input = self.concatenate_action_selectors([elem.initial_input for elem in inp], graph_offset)
        new_remove_input = self.concatenate_action_selectors([elem.remove_input for elem in inp], graph_offset)
        new_add_input = self.concatenate_action_selectors([elem.add_input for elem in inp], graph_offset)

        # New action targets
        new_initial_targets = self.concatenate_action_targets([elem.initial_target for elem in inp])
        new_remove_targets = self.concatenate_action_targets([elem.remove_target for elem in inp])
        new_add_targets = self.concatenate_action_targets([elem.add_target for elem in inp])

        # Stop targets
        new_stop_labels = np.concatenate([elem.stop_label for elem in inp])
        new_stop_mask = np.concatenate([elem.stop_mask for elem in inp])

        # num reactions
        num_reactions = sum([elem.num_reactions for elem in inp])

        # Pack back together
        op = ElectroTrainInput(graphs=new_graphs, reagents=new_reagents, initial_input=new_initial_input, remove_input=new_remove_input,
                          add_input=new_add_input, initial_target=new_initial_targets, remove_target=new_remove_targets,
                          add_target=new_add_targets, stop_label=new_stop_labels, stop_mask=new_stop_mask,
                               num_reactions=num_reactions)
        return op

    @staticmethod
    def concatenate_action_targets(targets: typing.List[graph_ds.ActionSelectorTarget]):
        # Labels
        new_labels = np.concatenate([elem.label_per_graph for elem in targets])

        # Masks
        new_masks = np.concatenate([elem.action_mask for elem in targets])

        # Put back together
        op = graph_ds.ActionSelectorTarget(new_labels, new_masks)
        return op

    @staticmethod
    def concatenate_action_selectors(action_selectors: typing.List[graph_ds.ActionSelectorGraphIds],
                                     graph_new_starting_locations: typing.List[int]):
        """
        Takes a list of  graph_ds.ActionSelectorGraphIds and creates a new one, taking into account that the
        indexing of the graphs and reagents has changed as we have concatenated them all together.
        """

        # Graph ids
        new_graphs_ids = np.concatenate([old_graph_id + offset for old_graph_id, offset
                          in zip([elem.graphs_ids for elem in action_selectors], graph_new_starting_locations)])

        # Actions
        old_actions = [elem.prev_action_per_graph for elem in action_selectors]
        if all([elem is None for elem in old_actions]):
            new_actions = None
        else:
            assert not any([elem is None for elem in old_actions])
            new_actions = np.concatenate(old_actions)

        # Reagents
        old_reagent_ids = [elem.reagent_context_ids for elem in action_selectors]
        if all([elem is None for elem in old_reagent_ids]):
            new_reagents = None
        elif all([elem == 0 for elem in old_reagent_ids]):
            new_reagents = np.arange(len(old_reagent_ids))
        else:
            raise RuntimeError("should only be one or None reagent per reaction with index 0")

        return graph_ds.ActionSelectorGraphIds(new_graphs_ids, new_actions, new_reagents)

