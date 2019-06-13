
import abc

import numpy as np
import torch
from rdkit import Chem

from .rdkit_ops import rdkit_general_ops

class MaskCreator(metaclass=abc.ABCMeta):
    def __init__(self, dev_str):
        self.dev_str = dev_str

    def mask_t(self, *args, **kwargs):
        return torch.from_numpy(self.mask_t(*args, **kwargs).to(self.dev_str))

    @abc.abstractmethod
    def mask_np(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.mask_np(*args, **kwargs)


class AddMaskCreator(MaskCreator):
    def mask_np(self, molecule: Chem.Mol, prev_atom_idx: int):
        num_atoms = rdkit_general_ops.get_num_atoms(molecule)
        mask = np.ones(num_atoms, dtype=bool)  # we can add to anywhere, except ...
        mask[prev_atom_idx] = 0   # ... to ourselves!
        return mask


class RemoveMaskCreator(MaskCreator):
    def mask_np(self, molecule:  Chem.Mol, prev_atom_idx: int, first_step_flag: bool):
        num_atoms = rdkit_general_ops.get_num_atoms(molecule)
        mask = np.zeros(num_atoms, dtype=bool)

        # We can only remove bonds that already exist, except ...
        available_bonds = rdkit_general_ops.get_other_atoms_attached_to_atom(molecule, prev_atom_idx)
        mask[list(available_bonds)] = True

        if first_step_flag:
            mask[prev_atom_idx] = True  # ... on the first step when we can also remove a self-bond

        return mask


class InitialActionMaskCreator(MaskCreator):
    def mask_np(self, molecule:  Chem.Mol):
        num_atoms = rdkit_general_ops.get_num_atoms(molecule)
        mask = np.ones(num_atoms, dtype=bool)  # we can start anywhere
        return mask


class StopMaskCreator(MaskCreator):
    def mask_np(self, num_steps_taken_upto_now: int, prev_was_self_bond_removal=False):
        """
        :param num_steps_taken_upto_now: can also think of this as the number of actions of the path that we have
        predicted. So before doing anything this should start at zero.
        """
        if num_steps_taken_upto_now == 1:
            # we cannot stop if we are yet to pick up a whole pair of electrons
            mask = np.array([0], dtype=bool)
        elif num_steps_taken_upto_now == 2 and prev_was_self_bond_removal:
            # we cannot stop after a self-bond removal.
            mask = np.array([0], dtype=bool)
        else:
            # but we can stop anywhere else
            mask = np.array([1], dtype=bool)
        return mask

    @staticmethod
    def all_time_steps_mask(*, action_list_ordered: list):
        num_stop_points = len(action_list_ordered) + 1
        # ^ the len on the list is the number of bond changes plus two. one extra for the final intitial graph, one
        # extra for the inital step in which we have not actually made any changes yet.
        mask = np.ones(num_stop_points, dtype=bool)
        mask[1] = False  # cannot stop after first pair
        if action_list_ordered[0] == action_list_ordered[1]:
            mask[2] = False  # cannot stop after self-bond removal -- these electrons must be placed somewhere.
        return mask


