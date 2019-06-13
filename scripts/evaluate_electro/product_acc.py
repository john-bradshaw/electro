
import argparse

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(4)
from rdkit import Chem

from rxn_steps.data.rdkit_ops import rdkit_general_ops
from rxn_steps.data import lef_uspto
from rxn_steps.data import transforms as r_trsfms


BOND_TYPE = [0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.QUADRUPLE]

VALENCE_CHECK = {'Na': 1, 'Li': 1, 'K': 1, 'Mg': 2, 'B': 3}
MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'S': 6, 'Sn': 4}
MAX_BONDS.update(VALENCE_CHECK)


def fix_endpoint(atom, verbose=False):
    explicit_bonds = int(sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()]))
    explicit_hs = atom.GetNumExplicitHs()
    atom_symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()

    if verbose:
        print("Atom is", atom_symbol, "with",
              charge, "charge,",
              explicit_bonds, "bonds,",
              explicit_hs, "explicit Hs")

    if atom_symbol in MAX_BONDS and explicit_bonds + explicit_hs > MAX_BONDS[atom_symbol]:
        if explicit_hs > 0:
            explicit_hs -= 1
            atom.SetNumExplicitHs(explicit_hs)
        else:
            charge += 1
            atom.SetFormalCharge(charge)
    if atom_symbol in VALENCE_CHECK:
        # Valence constraint isn't exactly the same as the max bonds constraint...
        if charge + explicit_hs + explicit_bonds > VALENCE_CHECK[atom_symbol]:
            if explicit_hs > 0:
                # if there are hydrogens, drop them
                explicit_hs -= 1
                atom.SetNumExplicitHs(explicit_hs)
            elif charge > 0:
                # remove an electron by stripping charge
                charge -= 1
                atom.SetFormalCharge(charge)


def make_rw_mol(source_smiles):
    """ Given a smiles, builds a RWMol
    function mostly derived from copy_edit_mol from https://github.com/wengong-jin/nips17-rexgen
    which had:
    MIT License

    Copyright (c) 2017 Wengong Jin, Connor W. Coley, Regina Barzilay, Tommi Jaakkola and Klavs F. Jensen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    mol = Chem.MolFromSmiles(source_smiles)
    Chem.Kekulize(mol)
    #     new_mol = Chem.RWMol(mol)
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def apply_edits(reactants, atom_path, verbose=False, atom_path_as_am_nums=True):
    """ Apply edits to a set of reactants, updating endpoints if needed.
        Also modifies start in case of a self-bond """
    new_mol = make_rw_mol(reactants)

    atom_to_idx = rdkit_general_ops.create_atom_map_indcs_map(new_mol)
    idx_to_atom = {v: k for k, v in atom_to_idx.items()}
    if atom_path_as_am_nums:
        atom_path_idx = [atom_to_idx[elem] for elem in atom_path]
    else:
        atom_path_idx = atom_path


    start_atom = None
    actions = list(zip(atom_path_idx[:-1], atom_path_idx[1:]))
    for i, (idx1, idx2) in enumerate(actions):
        remove_step = i % 2 == 0
        a1 = new_mol.GetAtomWithIdx(idx1)
        a2 = new_mol.GetAtomWithIdx(idx2)
        if verbose:
            print('%s:%d' % (a1.GetSymbol(), idx_to_atom[idx1]),
                  '%s:%d' % (a2.GetSymbol(), idx_to_atom[idx2]),
                  "REMOVE" if remove_step else "ADD")
        bond = new_mol.GetBondBetweenAtoms(idx1, idx2)
        count = 0 if bond is None else int(bond.GetBondTypeAsDouble())

        if i == 0:
            # grab initial atom for later
            start_atom = a1

        if remove_step:
            if idx1 != idx2:
                # Remove a bond
                new_mol.RemoveBond(idx1, idx2)
                if count - 1 > 0:
                    new_mol.AddBond(idx1, idx2, BOND_TYPE[count - 1])
            else:
                # This is a "self remove" -> try to strip explicit hydrogens
                hs = a1.GetNumExplicitHs()
                if hs > 0:
                    a1.SetNumExplicitHs(hs - 1)
                else:
                    # if there are no explicit hydrogens, change charge
                    charge = a1.GetFormalCharge()
                    if charge < 0:
                        a1.SetFormalCharge(charge + 1)
        else:  # Add step
            new_mol.RemoveBond(idx1, idx2)
            new_mol.AddBond(idx1, idx2, BOND_TYPE[count + 1])

    if len(actions) > 0:
        # fix start and end
        fix_endpoint(start_atom, verbose=verbose)
        fix_endpoint(a2, verbose=verbose)

    # Strip atom numbers
    pred_mol = new_mol.GetMol()
    for atom in pred_mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

    # make canonical smiles see lines 57-59 of from https://github.com/wengong-jin/nips17-rexgen
    pred_smiles = Chem.MolToSmiles(pred_mol).split('.')
    pred_mols = [Chem.MolFromSmiles(s) for s in pred_smiles]
    pred_smiles = [Chem.MolToSmiles(m) for m in pred_mols if m is not None]

    success = True
    if len(pred_smiles) < len(pred_mols):
        success = False

    return pred_smiles, success


def check_equality(candidate_edits, reactants, target_outcome, true_edits):
    if candidate_edits == true_edits:
        return True  # they match, including atom mapping
    pred = '.'.join(sorted(apply_edits(reactants, candidate_edits)[0]))
    return pred == target_outcome



def _get_data(use_val):
    tsfms = r_trsfms.TransformStrToBrokenDownParts()
    if use_val:
        print("Using USPTO LEF validation dataset")
        dataset = lef_uspto.LEFUspto(variant=lef_uspto.DataVariants.VAL, transform=tsfms)
    else:
        print("Using USPTO LEF test dataset")
        dataset = lef_uspto.LEFUspto(variant=lef_uspto.DataVariants.TEST, transform=tsfms)
    return dataset


def _unstr(line):
    topk_preds_ = line.split(';')

    def int_or_none(line_of_comma_seperated_ints):
        if len(line_of_comma_seperated_ints) > 0:
            res = list(map(int, line_of_comma_seperated_ints.split(',')))
        else:
            res = []
        return res
    return [int_or_none(elem) for elem in topk_preds_]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred", help="the path to the predictions ;-seperated with ,-sperated atom path.")
    parser.add_argument("--test_on_val", action="store_true", help="if set then will use validation dataset rather"
                        "than the test dataset")
    parser.add_argument("--run_first_x", default=0, type=int, help="number of test set to use, (0 means run all)")
    args = parser.parse_args()

    test_dataset = _get_data(args.test_on_val)
    num_to_run_on = len(test_dataset) if args.run_first_x == 0 else args.run_first_x
    print("Running on {} (dataset size is {})".format(num_to_run_on, len(test_dataset)))

    with open(args.pred, 'r') as fo:
        predicted_lines = fo.readlines()
        predicted_lines = [_unstr(x) for x in predicted_lines]

    print('n,top1,top2,top3,top5')
    for idx in range(num_to_run_on):
        test_ = test_dataset[idx]

        reactants = test_.reactants
        true_edits = [int(x) for x in test_.ordered_am_path]
        res, success = apply_edits(reactants, true_edits)
        assert success, "Error processing true path! Aborting"
        target_outcome = '.'.join(sorted(res))

        found = set()
        rk = 0
        incomplete = True
        predicted_ = predicted_lines[idx]
        for i in range(20):
            quick_accept = predicted_[i] == true_edits
            pred_smiles, success = apply_edits(reactants, predicted_[i])
            if success:
                this_pred = '.'.join(sorted(pred_smiles))
                if this_pred not in found:
                    rk += 1
                    if this_pred == target_outcome:
                        incomplete = False
                        break
                    assert not quick_accept
                    found.add(this_pred)
            else:
                rk += 1
        if incomplete:
            rk = 21

        print('%d,%d,%d,%d,%d' % (idx, rk == 1, rk <= 2, rk <= 3, rk <= 5))

