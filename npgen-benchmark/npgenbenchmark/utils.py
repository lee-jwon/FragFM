from math import sqrt
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

PROJECT_ROOT = Path(__file__).parents[1]

REPO_ID = "ICL-KAIST/NPGenBenchmark"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"


def download(file):
    """
    Download models from Hugging Face Hub to a local directory.
    """

    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    local_filename = Path(file).name
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type="model",
            local_dir=PROJECT_ROOT,
            local_dir_use_symlinks=False,
        )
        print(f"  Successfully downloaded to {downloaded_path}")
    except Exception as e:
        print(f"  Error downloading {file}: {e}")
    return


def isNAN(num):
    """Check if a number is NaN."""
    return num != num


def cosine_mat(x, y):
    """
    Calculate cosine similarity between two arrays.

    Args:
        x (np.array): First input array
        y (np.array): Second input array

    Returns:
        float: Cosine similarity
    """
    if np.sum(x**2) == 0 or np.sum(y**2) == 0:
        return 0
    else:
        return np.sum(x * y) / (sqrt(np.sum(x**2)) * sqrt(np.sum(y**2)))


def FP(smiles, radi):
    """
    Generate Morgan Fingerprints for a molecule.

    Args:
        smiles (str): SMILES representation of the molecule
        radi (int): Radius for Morgan Fingerprint

    Returns:
        tuple: Formula and binary fingerprints
    """
    binary = np.zeros((2048 * (radi)), int)
    formula = np.zeros((2048), int)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol_bi = {}
    for r in range(radi + 1):
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=r, bitInfo=mol_bi, nBits=2048
        )
        mol_bi_QC = []
        for i in mol_fp.GetOnBits():
            idx = mol_bi[i][0][0]
            radius_list = []
            num_ = len(mol_bi[i])
            for j in range(num_):
                if mol_bi[i][j][1] == r:
                    mol_bi_QC.append(i)
                    break

        if r == 0:
            for i in mol_bi_QC:
                formula[i] = len([k for k in mol_bi[i] if k[1] == 0])
        else:
            for i in mol_bi_QC:
                binary[(2048 * (r - 1)) + i] = len([k for k in mol_bi[i] if k[1] == r])

    return formula.reshape(1, 2048), binary.reshape(1, 4096)


def isglycoside(smiles):
    """
    Check if a molecule is a glycoside.

    Args:
        smiles (str): SMILES representation of the molecule

    Returns:
        bool: True if the molecule is a glycoside, False otherwise
    """
    sugar1 = Chem.MolFromSmarts("[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]")
    sugar2 = Chem.MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    )
    sugar3 = Chem.MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]"
    )
    sugar4 = Chem.MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]"
    )
    sugar5 = Chem.MolFromSmarts(
        "[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    )
    sugar6 = Chem.MolFromSmarts(
        "[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    )
    mol = Chem.MolFromSmiles(smiles)

    return any(
        [
            mol.HasSubstructMatch(sugar1),
            mol.HasSubstructMatch(sugar2),
            mol.HasSubstructMatch(sugar3),
            mol.HasSubstructMatch(sugar4),
            mol.HasSubstructMatch(sugar5),
            mol.HasSubstructMatch(sugar6),
        ]
    )
