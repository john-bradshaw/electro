
import enum
import typing
from os import path

from torch.utils import data


class DataVariants(enum.Enum):
    TRAIN = 'filtered_train.txt'
    VAL = 'filtered_valid.txt'
    TEST = 'filtered_test.txt'


class LEFUspto(data.Dataset):
    def __init__(self, variant: DataVariants, transform=None):
        data_location = path.abspath(path.join(__file__, path.pardir, path.pardir, path.pardir, 'lef_uspto'))
        file_to_read = path.join(data_location, variant.value)

        print(f"Reading file: {file_to_read}")

        with open(file_to_read, 'r') as fo:
            self.reaction_lines = fo.readlines()

        self.transform = transform

    def __getitem__(self, index):
        smiles = self.reaction_lines[index]
        rest, bond_changes = smiles.split()
        (reactants, products) = rest.split('>>')

        return_val: typing.Tuple[str] = (reactants, products, bond_changes)
        if self.transform is not None:
            return_val = self.transform(return_val)
        return return_val

    def __len__(self):
            return len(self.reaction_lines)









