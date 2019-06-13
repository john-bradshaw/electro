
import functools
import copy
import itertools
import collections

from rdkit import Chem

from graph_neural_networks.core import utils

from ..data.rdkit_ops import rdkit_general_ops
from ..data.rdkit_ops import rdkit_reaction_ops
from ..data.rdkit_ops import rdkit_featurization_ops
from ..data.rdkit_ops import chem_details
from ..data import mask_creators


class MaskHolders:
    def __init__(self, dev_str):
        self.stop_mask = mask_creators.StopMaskCreator(dev_str)
        self.initial_mask = mask_creators.InitialActionMaskCreator(dev_str)
        self.remove_mask = mask_creators.RemoveMaskCreator(dev_str)
        self.add_mask = mask_creators.AddMaskCreator(dev_str)


@functools.total_ordering
class ActionSequence(object):
    def __init__(self, initial_molecule, masks: MaskHolders, atom_feat_params=None):

        self.mask_compute_storage = masks
        self.actions_as_am: list = []   # stored in index format
        self.log_probs: list = []
        self._finished: bool = False

        self.intermediates_mols = {0: initial_molecule}
        self.action_am_to_idx_map = rdkit_general_ops.create_atom_map_indcs_map(initial_molecule)
        self.action_idx_to_am_map = utils.find_inverse_dict(self.action_am_to_idx_map)
        self.atom_feat_params = rdkit_featurization_ops.AtomFeatParams() if atom_feat_params is None else atom_feat_params

    @property
    def finished(self):
        return self._finished

    def mark_finished(self):
        self._finished = True

    @property
    def total_prob(self):
        return sum(self.log_probs)

    @property
    def prev_act_idx(self):
        return self.action_am_to_idx_map[self.actions_as_am[-1]]

    def __eq__(self, other):
        return other.total_prob == self.total_prob

    def __lt__(self, other):
        return self.total_prob < other.total_prob

    def __copy__(self):
        newone = type(self)(self.intermediates_mols[0], self.mask_compute_storage, self.atom_feat_params)
        newone.action_am_to_idx_map = self.action_am_to_idx_map
        newone.intermediates = copy.copy(self.intermediates_mols)
        newone._finished = self._finished
        newone.log_probs = copy.copy(self.log_probs)
        newone.actions_as_am = copy.copy(self.actions_as_am)
        return newone

    def __len__(self):
        return len(self.actions_as_am)

    def add_an_action(self, action_as_index, log_prob):
        action_as_am = self.action_idx_to_am_map[action_as_index]
        self.actions_as_am.append(action_as_am)
        self.log_probs.append(log_prob)

    def get_graph_and_masks(self, sequence_position: int):
        """
        returns the graph and masks for whether the model can stop and what actions are allowed for a particular
        sequence point. The data is not yet in PyTorch datastructures.
        """

        # Create the graph
        molecule = self.get_intermediate_molecule(sequence_position)
        graph = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(molecule, self.action_am_to_idx_map,
                                                                     params=self.atom_feat_params)

        # Create the stop mask
        if sequence_position > 1 and sequence_position % 2 == 0:
            # We have just done a remove step so if it was a self-bond remove then we should not be able to stop.
            prev_self_bond_removal = (self.action_am_to_idx_map[self.actions_as_am[sequence_position-1]] ==
                                      self.action_am_to_idx_map[self.actions_as_am[sequence_position-2]])
            stop_mask = self.mask_compute_storage.stop_mask(sequence_position, prev_self_bond_removal)
        else:
            stop_mask = self.mask_compute_storage.stop_mask(sequence_position, False)

        # Create the other mask
        if sequence_position == 0:
            # initial
            action_mask = self.mask_compute_storage.initial_mask(molecule)
        elif sequence_position % 2 == 1:
            # remove
            previous_action_idx = self.action_am_to_idx_map[self.actions_as_am[sequence_position-1]]
            action_mask = self.mask_compute_storage.remove_mask(molecule, previous_action_idx, sequence_position == 1)
        else:
            # add
            previous_action_idx = self.action_am_to_idx_map[self.actions_as_am[sequence_position-1]]
            action_mask = self.mask_compute_storage.add_mask(molecule, previous_action_idx)

        return graph, stop_mask, action_mask

    def get_intermediate_molecule(self, sequence_position: int):
        if not (len(self) >= sequence_position):
            raise RuntimeError("Trying to get a molecule further than the number of electron steps defines")

        if not (sequence_position  in self.intermediates_mols):
            # We then need to create the intermediate at this point.

            # we start with the closest molecule at a time step below:
            max_molecule_known_so_far = max(filter(lambda x: x < sequence_position, self.intermediates_mols.keys()))

            # Then we make the edits from this molcule (storing all intermediates)
            for action_number in range(max_molecule_known_so_far, sequence_position):
                prev_mol = self.intermediates_mols[action_number]

                if action_number == 0:
                    # it is an initial step so nothing changes
                    intermed = Chem.Mol(prev_mol)
                else:
                    # we are on an add or a remove step, we can find out which by parity:
                    step_type = chem_details.ElectronMode.REMOVE if action_number % 2 == 1 else chem_details.ElectronMode.ADD

                    # see where the electron has come from and where it is going to
                    prev_action_am = self.actions_as_am[action_number - 1]
                    action_am = self.actions_as_am[action_number]
                    action_idx = self.action_am_to_idx_map[action_am]
                    prev_action_idx = self.action_am_to_idx_map[prev_action_am]

                    if action_am == prev_action_am:
                        # This means  that we are on a self-bond step. This can only happen on certain occasions
                        # for which we check before carrying out
                        assert (step_type is chem_details.ElectronMode.REMOVE) and action_number == 1, \
                            "Trying to add a self bond or remove not on first step"
                        intermed = rdkit_reaction_ops.change_mol_atom(prev_mol, step_type, action_idx)
                    else:
                        # we are adding/removing an actual bond
                        intermed = rdkit_reaction_ops.change_mol_bond(prev_mol, step_type, (prev_action_idx, action_idx))

                self.intermediates_mols[action_number + 1] = intermed
                assert rdkit_general_ops.create_atom_map_indcs_map(intermed) == self.action_am_to_idx_map,\
                    "action map to index map changing!"
        return self.intermediates_mols[sequence_position]


