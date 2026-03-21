import os
import sys

import torch
from torch.utils.data import Dataset

from rdkit import Chem
sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))
try:
    from rdkit.Contrib.NP_Score import npscorer
except:
    from NP_Score import npscorer

from npgenbenchmark.utils import FP, isglycoside


class InferenceDataset(Dataset):
    def __init__(self, smiles_list, np_scorer):
        self.smiles_list = smiles_list
        self.np_scorer = np_scorer

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        fp_f, fp_b = FP(smi, 2)
        fp_f = torch.from_numpy(fp_f).flatten().float()
        fp_b = torch.from_numpy(fp_b).flatten().float()
        mol = Chem.MolFromSmiles(smi)
        npscore = npscorer.scoreMol(mol, self.np_scorer)
        return {
            "fp_f": fp_f,
            "fp_b": fp_b,
            "is_glycoside": isglycoside(smi),
            "smiles": smi,
            "npscore": npscore,
        }
