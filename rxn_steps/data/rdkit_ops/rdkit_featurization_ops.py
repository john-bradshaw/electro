import typing
import collections

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from graph_neural_networks.sparse_pattern import graph_as_adj_list

from . import chem_details
from ...misc import data_types


class AtomFeatParams:
    def __init__(self):
        self.atom_types = ['Ag', 'Al', 'Ar', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C',
                    'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'F',
                    'Fe', 'Ga', 'Ge', 'H', 'He', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La',
                    'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb',
                    'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se',
                    'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Te', 'Ti', 'Tl', 'V', 'W', 'Xe', 'Y',
                    'Yb', 'Zn', 'Zr']
        self.degrees = [0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,  10.]
        self.explicit_valences = [0., 1., 2., 3., 4., 5., 6., 7., 8., 10., 12., 14.]
        self.atom_feature_length = len(self.atom_types) + len(self.degrees) + len(self.explicit_valences) + 8
        self.bond_names = ['single', 'double', 'triple']
        self.num_bond_types = len(self.bond_names)

    def get_bond_name(self, bond):
        bt = bond.GetBondType()
        return {
            Chem.rdchem.BondType.SINGLE: 'single',
            Chem.rdchem.BondType.DOUBLE: 'double',
            Chem.rdchem.BondType.TRIPLE: 'triple',
        }[bt]


def mol_to_atom_feats_and_adjacency_list(mol: AllChem.Mol, atom_map_to_index_map=None,
                                         params: AtomFeatParams=None):
    """
    :param atom_map_to_index_map: if you pass this in it will use the defined indices for each atom. Otherwise will use
    rdkit default indexing.
    """
    params = AtomFeatParams() if params is None else params
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)

    node_feats = np.zeros((num_atoms, params.atom_feature_length), dtype=np.float32)
    idx_to_atom_map = np.zeros(num_atoms, dtype=np.float32)

    if atom_map_to_index_map is None:
        # then we will create this map
        atom_map_to_index_map = {}
        use_supplied_idx_flg = False
    else:
        # we will use the mapping given
        use_supplied_idx_flg = True
        assert set(atom_map_to_index_map.values()) == set(range(len(atoms))), \
            "if give pre supplied ordering it must be the same size as the molecules trying to order"

    # First we will create the atom features and the mappings
    for atom in atoms:
        props = atom.GetPropsAsDict()
        am = props['molAtomMapNumber']  # the atom mapping in the file
        if use_supplied_idx_flg:
            idx = atom_map_to_index_map[am]
        else:
            idx = atom.GetIdx()  # goes from 0 to A-1
            atom_map_to_index_map[am] = idx
        idx_to_atom_map[idx] = am
        atom_features = get_atom_features(atom, params)
        node_feats[idx, :] = atom_features

    # Now we will go through and create the adjacency lists
    adjacency_lists = {k: [] for k in params.bond_names}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        props_b = begin.GetPropsAsDict()
        props_e = end.GetPropsAsDict()
        am_b = props_b['molAtomMapNumber']
        am_e = props_e['molAtomMapNumber']
        ix_b = atom_map_to_index_map[am_b]
        ix_e = atom_map_to_index_map[am_e]

        bond_name = params.get_bond_name(bond)
        adjacency_lists[bond_name].append((ix_b, ix_e))

    # Finally we pack all the results together
    res = graph_as_adj_list.GraphAsAdjList(node_feats, {k: np.array(v).T for k, v in adjacency_lists.items()},
                                     np.zeros(node_feats.shape[0], dtype=data_types.INT))

    return res


def create_empty_graph(params: AtomFeatParams=None):
    params = AtomFeatParams() if params is None else params
    node_feats = np.zeros((5, params.atom_feature_length), dtype=np.float32)
    res = graph_as_adj_list.GraphAsAdjList(node_feats, {k:np.array([]) for k in params.bond_names},
                                     np.zeros(node_feats.shape[0], dtype=data_types.INT))
    return res


def get_atom_features(atom, params: AtomFeatParams=None):
    params = AtomFeatParams() if params is None else params

    # the number of features and their Indices are shown in comments, although be cautious as these
    # may change as we decide what features to give.
    return np.array(onek_encoding_unk(atom.GetSymbol(), params.atom_types, 0)  # 72 [0-71]
                    + onek_encoding_unk(atom.GetDegree(), params.degrees, 0)  # 9  [72-80]
                    + onek_encoding_unk(atom.GetExplicitValence(),
                                                         params.explicit_valences, 0)  # 12 [81-92]
                    + onek_encoding_unk(atom.GetHybridization(),
                                                         [Chem.rdchem.HybridizationType.SP,
                                                          Chem.rdchem.HybridizationType.SP2,
                                                          Chem.rdchem.HybridizationType.SP3, 0], 1)  # 4 [93-96]
                    + [atom.GetTotalNumHs()]  # 1 [97]
                    + [chem_details.electroneg[atom.GetSymbol()]]  # 1 [98]
                    + [atom.GetAtomicNum()]  # 1 [99]
                    + [atom.GetIsAromatic()], dtype=np.float32)  # 1 [100]


def get_bond_feats(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE],
        dtype=np.float32)


def onek_encoding_unk(x, allowable_set, if_missing_set_as_last_flag):
    if x not in set(allowable_set):
        if if_missing_set_as_last_flag:
            x = allowable_set[-1]
        else:
            raise RuntimeError
    return list(map(lambda s: x == s, allowable_set))


def get_morgan_finerprints(molecule, radius=4, num_bits=200):
    bv = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, num_bits)
    arr = np.zeros(num_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

