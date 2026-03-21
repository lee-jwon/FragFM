import re
from copy import deepcopy

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def reconstruct_to_rdmol(h, e_index, e, is_relaxed=False, get_largest=True, fix=False):
    mol = Chem.RWMol()
    atom_indices = []
    for atomic_num in h:
        atom = Chem.Atom(int(atomic_num))
        atom_idx = mol.AddAtom(atom)
        atom_indices.append(atom_idx)
    for i in range(e_index.shape[1]):
        atom1_idx = int(e_index[0, i])
        atom2_idx = int(e_index[1, i])
        bond_type_val = e[i]
        if bond_type_val == 0:
            pass
        elif bond_type_val == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_type_val == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_type_val == 3:
            bond_type = Chem.BondType.TRIPLE
        elif bond_type_val == 4:
            bond_type = Chem.BondType.AROMATIC
        else:
            raise ValueError(f"Unknown bond type value: {bond_type_val}")
        mol.AddBond(atom1_idx, atom2_idx, bond_type)
        if bond_type_val == 4:
            # Mark the atoms and bond as aromatic
            mol.GetAtomWithIdx(atom1_idx).SetIsAromatic(True)
            mol.GetAtomWithIdx(atom2_idx).SetIsAromatic(True)
            bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            bond.SetIsAromatic(True)

        # relax valency (charged molecules)
        if is_relaxed:
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)

    if fix:
        ori_mol = deepcopy(mol)
        mol, no_cor = correct_mol(mol)

    if get_largest:
        # get largest connected component
        mol = valid_mol_can_with_seg(mol, largest_connected_comp=get_largest)
    else:
        mol = mol.GetMol()

    assert mol is not None
    mol = Chem.RemoveHs(mol)

    return mol


def mol_to_atomic_number_matrix(mol):
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atomic_number_matrix = np.array(atomic_numbers)
    return atomic_number_matrix


def mol_to_edge_index_and_type(mol):
    edge_index = []
    edge_type = []
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        edge_index.append([atom1_idx, atom2_idx])
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            edge_type.append(1)  # Single bond
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            edge_type.append(2)  # Double bond
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            edge_type.append(3)  # Triple bond
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            edge_type.append(4)  # Aromatic bond
    edge_index = np.array(edge_index).T  # Transpose to make it a 2xN matrix
    edge_type = np.array(edge_type)
    return edge_index, edge_type


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    # function from GDSS
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [(s, len(s)) for s in sm.split(".")]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


bond_decoder = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}


def correct_mol(m):
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (
                        b.GetIdx(),
                        int(b.GetBondType()),
                        b.GetBeginAtomIdx(),
                        b.GetEndAtomIdx(),
                    )
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence
