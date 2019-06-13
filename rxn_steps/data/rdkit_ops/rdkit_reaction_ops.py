import functools
import typing
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem

from . import chem_details
from .rdkit_general_ops import get_atom_map_nums
from .rdkit_general_ops import create_atom_map_indcs_map
from .rdkit_general_ops import get_bond_double_between_atom_mapped_atoms
from .rdkit_general_ops import NUM_TO_BOND



def split_reagents_out_from_reactants_and_products(reactant_all_str: str, product_all_str: str,
                                                   actions_unordered_am: list) -> typing.Tuple[str, str, str]:
    """
    :param reactant_all_str: SMILES string of all reactants -- individual reactants seperated by dots.
    :param product_all_str: SMILES string of all products -- individual reactants seperated by dots.
    :param actions_unordered_am: list of atoms involved in the electron path
    :return:
    """
    reactants_str = reactant_all_str.split('.')
    products_str = product_all_str.split('.')

    product_smiles_set = set(products_str)
    products_to_keep = set(products_str)
    product_atom_map_nums = functools.reduce(lambda x, y: x | y, (get_atom_map_nums(prod) for prod in products_str))
    actions_atom_map_nums = set(actions_unordered_am)

    reactants = []
    reagents = []
    for candidate_reactant in reactants_str:
        atom_map_nums = get_atom_map_nums(candidate_reactant)

        # a) any atoms in products
        in_product = list(product_atom_map_nums & atom_map_nums)
        # b) any atoms in reaction center
        in_center = list(set(actions_atom_map_nums & atom_map_nums))

        if (len(in_product) == 0) and (len(in_center) == 0):  # this is a reagent
            reagents.append(candidate_reactant)
        else:
            if candidate_reactant in product_smiles_set:
                #TODO ^ this should be canoncical comparison instead.
                reagents.append(candidate_reactant)
                products_to_keep -= set(candidate_reactant)  # remove it from the products too.
            else:
                reactants.append(candidate_reactant)

    product_all_str = '.'.join(products_to_keep)
    return '.'.join(reactants), '.'.join(reagents), product_all_str


def change_mol_bond(mol: AllChem.Mol, diff_mode: chem_details.ElectronMode, bond_to_change_indcs: typing.Tuple[int, int]):
    """
    Change a molecule by adding or removing a pair of electrons from a bond.
    """
    ed_mol = Chem.RWMol(mol)
    exists = mol.GetBondBetweenAtoms(bond_to_change_indcs[0], bond_to_change_indcs[1])

    # Either we are reducing the number of pairs of electrons in the bond by one.
    if diff_mode is chem_details.ElectronMode.REMOVE:
        # a. we first remove the bond:
        ed_mol.RemoveBond(bond_to_change_indcs[0], bond_to_change_indcs[1])

        # b. we then (if it had more than one pair of electrons) add it back with one less pair of electrons than before:
        if exists:
            bt_d = exists.GetBondTypeAsDouble()
            if bt_d - 1 != 0:
                new_bt = NUM_TO_BOND[bt_d - 1]
                ed_mol.AddBond(bond_to_change_indcs[0], bond_to_change_indcs[1], order=new_bt)

    # Or we are increasing the number of pairs of electrons in the bond by one.
    elif diff_mode is chem_details.ElectronMode.ADD:
        if exists:
            # a. if it already exists we remove it and add it back with an extra pair of electrons
            bt_d = exists.GetBondTypeAsDouble()
            if bt_d + 1 not in NUM_TO_BOND:
                new_bt = NUM_TO_BOND[bt_d]  # if already at maximum we leave it as it is (we do not deal with aromatic)
            else:
                new_bt = NUM_TO_BOND[bt_d + 1]
            ed_mol.RemoveBond(bond_to_change_indcs[0], bond_to_change_indcs[1])
            ed_mol.AddBond(bond_to_change_indcs[0], bond_to_change_indcs[1], order=new_bt)
        else:
            # b. if it does not exist then we create a single bond.
            ed_mol.AddBond(*bond_to_change_indcs, order=NUM_TO_BOND[1])
    else:
        raise RuntimeError("Invalid mode: {}".format(diff_mode))
    new_mol = ed_mol.GetMol()
    return new_mol


def change_mol_atom(mol: AllChem.Mol, diff_mode: chem_details.ElectronMode, atom_to_change_idx):
    """
    change a molecule by removing an electron from an atom -- ie break a "self-bond"
    """
    ed_mol = Chem.RWMol(mol)
    if diff_mode is chem_details.ElectronMode.ADD:
        raise NotImplementedError("Adding electron to self not currently implemented")

    # If remove we remove a pair of electrons from the atom. As one electron will be shared as part of a new bond we
    # only increase the postive charge by one:
    elif diff_mode is chem_details.ElectronMode.REMOVE:
        current_atom = ed_mol.GetAtomWithIdx(atom_to_change_idx)
        current_atom.SetFormalCharge(current_atom.GetFormalCharge() + 1)
    else:
        raise RuntimeError("Diff mode {} not supported".format(diff_mode))
    new_mol = ed_mol.GetMol()
    return new_mol


def is_it_alternating_add_and_remove_steps(reactant_mol, product_mol, actions_am):
    previous = None
    react_am_to_indcs_map = create_atom_map_indcs_map(reactant_mol)
    prod_am_to_indcs_map = create_atom_map_indcs_map(product_mol)

    for begin_am, end_am in zip(actions_am[:-1], actions_am[1:]):
        if begin_am == end_am:
            # This can only be a self bond removal
            change = -1
        else:
            new_bond = get_bond_double_between_atom_mapped_atoms(product_mol, begin_am, end_am, prod_am_to_indcs_map)
            orig_bond = get_bond_double_between_atom_mapped_atoms(reactant_mol, begin_am, end_am, react_am_to_indcs_map)
            change = new_bond - orig_bond
        cond1 = change not in {1, -1}
        cond2 = (previous is not None) and (change + previous != 0)
        if cond1 or cond2:
            return False
        previous = change
    return True

def is_sub_mol_consistent_with_super(sub_mol, super_mol, raise_on_aromatic_changes=False):
    """
    Checks that the sub molecule is consistent with the super molecule.
    Ie within the sub molecule the bonds should be connected
    the same way as in the supermolecule and only disconnected parts should be in the super molecule.
    """
    sub_mol_atom_mappings_to_indcs = create_atom_map_indcs_map(sub_mol)
    ams_in_sub_mol = set(sub_mol_atom_mappings_to_indcs.keys())

    # Our first check is that all bonds in the sub molecule exist in the same form in the super molecule
    for bnd in super_mol.GetBonds():
        begin_am = bnd.GetBeginAtom().GetPropsAsDict()['molAtomMapNumber']
        end_am = bnd.GetEndAtom().GetPropsAsDict()['molAtomMapNumber']
        intersection = {begin_am, end_am} & ams_in_sub_mol

        # 1. If both atoms exist in the sub molecule too then they should be connected in the same way:
        if len(intersection) == 2:
            bnd_type = bnd.GetBondType()
            sub_bond = sub_mol.GetBondBetweenAtoms(sub_mol_atom_mappings_to_indcs[begin_am],
                                                     sub_mol_atom_mappings_to_indcs[end_am])

            # 1a Bond in super molecule but none in the sub molecule => Inconsistent:
            if sub_bond is None:
                return False

            # 1b Bond in super molecule and in the sub molecule but different type:
            if not sub_bond.GetBondType() == bnd_type:

                # 1b,i If part of a ring then we could just have kekulized differently => Maybe inconsistent
                # so only return False if flag set:
                if bnd.GetBeginAtom().GetIsAromatic() and bnd.GetEndAtom().GetIsAromatic():
                    warnings.warn("We potentially are having changes in kekulization.")
                    if raise_on_aromatic_changes:
                        return False

                # 1b,ii Not part of a ring and bond does not match => Inconsistent
                else:
                    return False

        # The super molecule(s) is connected up to more atoms than the sub molecule so graph is larger => Inconsistent
        elif len(intersection) == 1:
            return False

        #  There is a bond between two atoms not in the submolecule and so we have nothing to check against.
        else:
            pass

    return True

