import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

import fragfm.rBRICS_public as rBRICS
from fragfm.utils.mol_ops import *


def merge_mols(mol_list: list[Chem.Mol]) -> Chem.Mol:
    merged_mol = mol_list[0]
    for mol in mol_list[1:]:
        merged_mol = Chem.CombineMols(merged_mol, mol)
    return merged_mol


def get_connected_pairs(adj_matrix: np.ndarray) -> list[tuple[int, int]]:
    connected_pairs = []
    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                connected_pairs.append((i, j))
    return connected_pairs


def get_adjacency_matrix_from_mol(mol: Chem.Mol) -> np.ndarray:
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Since it's undirected, set the reverse as well
    return adjacency_matrix


def get_atom_indices_from_bond_indices(
    mol: Chem.Mol, bond_indices: np.ndarray
) -> np.ndarray:
    atom_indices = []
    for bond_idx in bond_indices:
        bond = mol.GetBondWithIdx(int(bond_idx))  # Get the bond by its index
        atom1_idx = bond.GetBeginAtomIdx()  # Atom index 1
        atom2_idx = bond.GetEndAtomIdx()  # Atom index 2
        atom_indices.append([atom1_idx, atom2_idx])
    return np.array(atom_indices).astype(int).T


def get_brics_bond_indices(mol: Chem.Mol) -> np.ndarray:
    decomposable_bonds = BRICS.FindBRICSBonds(mol)
    bond_indices = []
    for bond_info in decomposable_bonds:
        atom1, atom2 = bond_info[0]  # Get the atoms forming the bond
        bond = mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            bond_idx = bond.GetIdx()  # Get the bond index
            bond_indices.append(bond_idx)
    return np.array(bond_indices).astype(int)


def get_rbrics_bond_indices(mol: Chem.Mol) -> np.ndarray:
    decomposable_bonds = rBRICS.FindrBRICSBonds(mol)
    bond_indices = []
    for bond_info in decomposable_bonds:
        atom1, atom2 = bond_info[0]  # Get the atoms forming the bond
        bond = mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            bond_idx = bond.GetIdx()  # Get the bond index
            bond_indices.append(bond_idx)
    return np.array(bond_indices).astype(int)
