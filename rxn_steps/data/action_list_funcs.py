
import itertools

import numpy as np

from . import lef_consistency_errors as exceptions
from .rdkit_ops import chem_details
from .rdkit_ops import rdkit_general_ops
from .rdkit_ops import rdkit_reaction_ops

class OrderUndefinedByDataError(RuntimeError):
    pass


def order_actions_am(actions_am: list, reactant_mol, product_mol):
    """
    Extract atom walk from the pairs of atoms that have changed and then order it correctly.
    """
    # 1. We collect the atom map ids for the current first and last atoms in the electron walk
    current_start_atom_as_am = actions_am[0]
    current_end_atom_as_am = actions_am[-1]
    atoms_interested_in = {current_start_atom_as_am, current_end_atom_as_am}

    # 2. We then collect element type and charge details for these atoms in the reactants and products
    # (but they may not be detailed in the products)
    details_at_start = rdkit_general_ops.get_atoms_names_charge_and_h_count(reactant_mol, atoms_interested_in)
    details_at_end = rdkit_general_ops.get_atoms_names_charge_and_h_count(product_mol, atoms_interested_in)

    # 3. We then look at the change of charge at the ends of the walk.
    try:
        # We first look at the current last atom in the walk.
        charge_change = details_at_end[current_end_atom_as_am][1] - details_at_start[current_end_atom_as_am][1]
        hyd_change = details_at_end[current_end_atom_as_am][2] - details_at_start[current_end_atom_as_am][2]
        flip_flag = is_flip_in_action_ordering_required(charge_change, hyd_change, True)
        # TODO: we could consider checking the other end of the actions is consistent with this (if we have this \
        # information)
    except (KeyError, OrderUndefinedByDataError) as ex:
        # We do not have details of the current last atom state in the products
        try:
            # So we look instead at the current first atom state in the products
            charge_change = details_at_end[current_start_atom_as_am][1] - details_at_start[current_start_atom_as_am][1]
            hyd_change = details_at_end[current_start_atom_as_am][2] - details_at_start[current_start_atom_as_am][2]
            flip_flag = is_flip_in_action_ordering_required(charge_change, hyd_change, False)
        except (KeyError, OrderUndefinedByDataError) as ex:
            # We don't have this either! So we just use electronegativity of the elements involved.
            # This should give a sensible order for electron movement.
            electroneg_current_start = chem_details.electroneg[details_at_start[current_start_atom_as_am][0]]
            electroneg_current_end = chem_details.electroneg[details_at_start[current_end_atom_as_am][0]]
            currently_in_correct_order = electroneg_current_end - electroneg_current_start > 0
            flip_flag = not currently_in_correct_order
            # if electonegativity is equal we do not know so shall just leave it as it is.

    # 4. We reverse ordering if required.
    if flip_flag:
        actions_am = list(reversed(actions_am))

    return actions_am


def is_flip_in_action_ordering_required(change_in_atom_charge: int, change_in_hydrogens: int, atom_is_currently_last_action: bool):
    """
    Work out whether electron path is currently in the correct order based on details about changes between both ends.
    """

    total_effective_charge_change = change_in_atom_charge - change_in_hydrogens

    if total_effective_charge_change == -1:
        return not atom_is_currently_last_action
        # ^ the atom at the final action should lose charge as the electrons arrive
    elif total_effective_charge_change == 1:
        return atom_is_currently_last_action
        # ^ The atom at the initial action should gain charge as the electrons move away.
    else:
        raise OrderUndefinedByDataError("One of the reaction ends has neither lost nor gained charge ("
                                             "or implicit Hydrogens).")


def actions_am_from_pairs(atom_pairs: np.ndarray, consistency_check=True) -> list:
    """
    This is like playing a game of dominoes. We have a series of pairs and we try to find one line.
    nb we do not care about the ordering of this list at the moment.
    :param pairs: np array shape (num_bonds changed, 2) contains the start and end atoms for every bond changed.
    :return: actions related to atom mappings
    """
    atom_list = atom_pairs.reshape(-1)
    unique, counts = np.unique(atom_list, return_counts=True)
    if consistency_check and np.sum(counts > 2):
        raise exceptions.NonLinearTopologyException("Atoms {} are involved in more than one walk.".format(unique[counts > 2].tolist()))

    if consistency_check and np.sum(counts == 1) != 2:
        raise exceptions.NonLinearTopologyException("Potentially multiple walk from the atoms".format(unique[counts == 1].tolist()))

    start_end = np.where(counts == 1)[0]
    start = unique[start_end[0]]

    # This next bit is like lining up dominoes into one chain.
    actions_am = [start]
    def find_next(current_list, remaining_pairs):
        for pair in remaining_pairs:
            if current_list[-1] in pair:
                remaining_pairs.remove(pair)
                pair.remove(current_list[-1])
                current_list += list(pair)
                return find_next(current_list, remaining_pairs)
    find_next(actions_am, [set(item) for item in atom_pairs])
    return actions_am
