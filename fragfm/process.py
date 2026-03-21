import logging
import pickle
from copy import deepcopy

import lmdb
import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from tqdm import tqdm

import fragfm.rBRICS_public as rBRICS

generator = rdNormalizedDescriptors.RDKit2DNormalized()
generator.columns  # list of tuples:  (descriptor_name, numpytype) ...


from fragfm.utils.graph_ops import (
    adje_to_sparse_edge,
    e_index_e_to_adje,
    get_independent_nodes_from_adj,
    mask_pairs_from_adje,
    slice_array,
    sparse_edge_to_fully_connected_edge,
)
from fragfm.utils.mol_decompose_ops import (
    get_atom_indices_from_bond_indices,
    get_brics_bond_indices,
    get_rbrics_bond_indices,
)
from fragfm.utils.mol_ops import (
    mol_to_atomic_number_matrix,
    mol_to_edge_index_and_type,
    reconstruct_to_rdmol,
)


def get_descriptor_from_smiles(smiles: str):
    results = generator.process(smiles)
    processed, features = results[0], results[1:]
    assert processed
    features = (
        features[:39]
        + features[40:41]
        + features[42:43]
        + features[44:45]
        + features[46:]
    )
    features = np.array(features)
    assert not np.any(np.isnan(features))
    return features


def create_lmdb_dataset(data, lmdb_path, map_size=3e9):
    env = lmdb.open(lmdb_path, map_size=int(map_size))
    with env.begin(write=True) as txn:
        for sample in tqdm(data, disable=True):
            txn.put(str(sample["key"]).encode(), pickle.dumps(sample))
    env.close()


def process_data(data):
    samples = []
    for sample in tqdm(data, disable=False):
        ori_sample = deepcopy(sample)
        try:
            new_sample = process_sample(sample)
            sample.update(new_sample)
            samples.append(sample)
            # print('success', ori_sample)
        except Exception as e:
            print(f"{sample} {e}", flush=True)
            pass
    return samples


if False:

    def process_data(data):
        samples = []
        for sample in tqdm(data):
            print(sample)
            if True:
                new_sample = process_sample(sample)
                sample.update(new_sample)
                samples.append(sample)
        return samples


def process_sample(sample):
    smi = sample["smi"]
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    out = {}
    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)

    # process
    h = mol_to_atomic_number_matrix(mol_h)
    e_index, e = mol_to_edge_index_and_type(mol_h)
    e_index, e = sparse_edge_to_fully_connected_edge(e_index, e, n_node=h.shape[0])
    adje = e_index_e_to_adje(e_index, e, n=h.shape[0])

    # get decomposed graph
    if sample["decomp_method"] == "brics":
        d_bond_index = get_brics_bond_indices(mol_h)
    elif sample["decomp_method"] == "rbrics":
        temp_d_bond_index = get_rbrics_bond_indices(mol_h)
        temp_d_atom_idx = get_atom_indices_from_bond_indices(mol_h, temp_d_bond_index)
        temp_decomp_adje = mask_pairs_from_adje(adje, temp_d_atom_idx)
        temp_frag_node_groups = get_independent_nodes_from_adj(temp_decomp_adje)
        if temp_d_atom_idx.shape[0] == 0:
            d_bond_index = np.array([])
        else:
            valid_pair_idx = _filter_all_valid_pairs(
                temp_d_atom_idx, temp_frag_node_groups
            )
            d_bond_index = temp_d_bond_index[valid_pair_idx]
    else:
        raise NotImplementedError

    if d_bond_index.shape[0] == 0:
        is_single_frag = True
    else:
        is_single_frag = False

    if not is_single_frag:
        d_index = get_atom_indices_from_bond_indices(mol_h, d_bond_index)
        decomp_adje = mask_pairs_from_adje(adje, d_index)
        junction_atom_indexs = d_index.ravel()
        h_junction_count = np.array(
            [
                np.count_nonzero(junction_atom_indexs == x)
                for x in list(range(h.shape[0]))
            ]
        )
    else:
        decomp_adje = adje
        h_junction_count = np.array([0 for _ in list(range(h.shape[0]))])

    # decompose to fragments (pseudo diagonal terms)
    frag_node_idxs = get_independent_nodes_from_adj(decomp_adje)

    # get reordering map of atom in each fragments to make it canonical
    reorder_match_list = []
    smi_list = []
    for frag_node_idx in frag_node_idxs:
        frag_node_idx = np.array(frag_node_idx)
        frag_h = h[frag_node_idx]
        frag_adje = slice_array(adje, frag_node_idx, frag_node_idx)
        frag_e_index, frag_e = adje_to_sparse_edge(frag_adje)
        if len(frag_e_index) == 0:
            frag_e_index = np.empty((2, 0)).astype(int)
        frag_h_junction_count = h_junction_count[frag_node_idx]

        # check match
        if sample["data_type"] in ["debug", "moses", "npgen"]:
            smi, match = get_canonical_reordering_map_for_frag(
                frag_h, frag_h_junction_count, frag_e_index, frag_e
            )
            reorder_match_list.append(frag_node_idx[match])
            smi_list.append(smi)
        elif sample["data_type"] in ["guacamol", "zinc250k"]:
            smi, match = get_canonical_reordering_map_for_frag_relaxed(
                frag_h, frag_h_junction_count, frag_e_index, frag_e
            )
            reorder_match_list.append(frag_node_idx[match])
            smi_list.append(smi)
        else:
            raise NotImplementedError

    # check fragment processed
    for smi in smi_list:
        if sample["data_type"] in ["debug", "moses", "npgen"]:
            process_frag(smi, is_relax=False)
        elif sample["data_type"] in ["guacamol", "zinc250k"]:
            process_frag(smi, is_relax=True)

    # convert it to global node ranking
    n_frag_atoms = np.array([len(x) for x in reorder_match_list])
    global_match = np.concatenate(reorder_match_list)

    # get fragment level graph informations
    out["frag_smi_list"] = smi_list
    out["n_frag"] = len(smi_list)
    out["n_frag_atom"] = n_frag_atoms
    out["h_frag_batch"] = np.concatenate(
        [np.full(count, idx) for idx, count in enumerate(n_frag_atoms)]
    )
    out["h_in_frag_label"] = np.array(
        [i for count in n_frag_atoms for i in range(count)]
    )
    assert len(out["h_in_frag_label"]) == len(h), (
        f"{len(h)}, {len(out['h_in_frag_label'])}"
    )

    # convert the data by <global match> reordering
    out["h"] = h[global_match]
    out["h_junction_count"] = h_junction_count[global_match]
    adje = adje[np.ix_(global_match, global_match)]
    decomp_adje = decomp_adje[np.ix_(global_match, global_match)]
    frag_index = [[i for _ in range(x)] for i, x in enumerate(n_frag_atoms)]
    out["frag_index"] = np.array([item for sublist in frag_index for item in sublist])
    out["e_index"], out["e"] = adje_to_sparse_edge(adje)
    out["decomp_e_index"], out["decomp_e"] = adje_to_sparse_edge(
        decomp_adje
    )  # 1: single ~ 4: aromatic

    # make fragment level graph (all edge types are the same, only index needed)
    k = len(np.unique(out["frag_index"]))
    coarse_adje = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            if i != j:
                group_i_indices = np.where(out["frag_index"] == i)[0]
                group_j_indices = np.where(out["frag_index"] == j)[0]
                if np.any(adje[np.ix_(group_i_indices, group_j_indices)]):
                    coarse_adje[i, j] = 1
    out["coarse_e_index"], _ = adje_to_sparse_edge(coarse_adje)
    if len(out["coarse_e_index"]) == 0:
        out["coarse_e_index"] = np.empty((2, 0)).astype(int)

    # make extended matrix of coarse adje in the size of fine graph (block matrix)
    filled_adj_matrix = np.zeros((out["h"].shape[0], out["h"].shape[0]), dtype=int)
    count = 1
    for i in range(k):
        for j in range(k):
            group_i_indices = np.where(out["frag_index"] == i)[0]
            group_j_indices = np.where(out["frag_index"] == j)[0]
            if coarse_adje[i, j] == 1 and i < j:
                filled_adj_matrix[np.ix_(group_i_indices, group_j_indices)] = (
                    coarse_adje[i, j] * count
                )
                count += 1
    filled_adj_matrix = filled_adj_matrix + filled_adj_matrix.T

    # make data for autoencoder to predict
    j2j_mask = np.zeros((out["h"].shape[0], out["h"].shape[0]), dtype=int)
    for i in range(out["h"].shape[0]):
        for j in range(out["h"].shape[0]):
            if out["h_junction_count"][i] != 0 and out["h_junction_count"][j] != 0:
                j2j_mask[i, j] = 1
    to_pred_mask = ((j2j_mask == 1) & (filled_adj_matrix != 0)).astype(int)
    to_pred_ans = to_pred_mask * adje
    out["ae_to_pred_index"], ae_to_pred_ans_p = adje_to_sparse_edge(
        to_pred_mask + to_pred_ans
    )
    out["ae_to_pred"] = ae_to_pred_ans_p - 1
    if len(out["ae_to_pred"]) == 0:
        out["ae_to_pred_index"] = np.empty((2, 0)).astype(int)
        out["ae_to_pred"] = np.empty(0).astype(int)
    return out


def process_frag(frag_smi, is_relax=False):
    frag_smi = Chem.MolToSmiles(Chem.MolFromSmiles(frag_smi))
    out = {}
    mol = Chem.MolFromSmiles(frag_smi)
    mol_h = Chem.AddHs(mol)
    h = mol_to_atomic_number_matrix(mol_h)
    e_index, e = mol_to_edge_index_and_type(mol_h)
    adje = e_index_e_to_adje(e_index, e, n=h.shape[0])

    # get h_junction_count
    zer_mask = h == 0
    h_juncion_count = (adje == 1).astype(int).dot(zer_mask.astype(int))

    h = h[zer_mask == False]
    h_juncion_count = h_juncion_count[zer_mask == False]
    adje = slice_array(adje, zer_mask == False, zer_mask == False)
    e_index, e = adje_to_sparse_edge(adje)

    # checker
    if not is_relax:
        can_frag_smi, sample = get_canonical_reordering_map_for_frag(
            h, h_juncion_count, e_index, e
        )
    else:
        can_frag_smi, sample = get_canonical_reordering_map_for_frag_relaxed(
            h, h_juncion_count, e_index, e
        )
    assert frag_smi == can_frag_smi

    out["g"] = get_descriptor_from_smiles(frag_smi)
    out["h"] = h
    out["h_junction_count"] = h_juncion_count
    out["e_index"] = e_index
    out["e"] = e
    return out


def add_atom_to_junction(h, h_junction_count, e_index, e, new_atom_idx=0):
    """
    0 indicates *, and 1 indicates hydrogen (H)
    """
    for i, jc in enumerate(h_junction_count):
        for _ in range(jc):
            new_idx = h.shape[0]
            h = np.append(h, new_atom_idx)
            e = np.append(e, 1)  # single bond added
            e_index = np.concatenate(
                [e_index, np.array([i, new_idx]).reshape(2, 1)], axis=1
            )
    return h, e_index, e


def remove_juntion_atom(h, adje):
    to_leave_idx = np.where(h != 0)[0]
    adje = slice_array(adje, to_leave_idx, to_leave_idx)
    h = h[to_leave_idx]
    return h, adje


def get_canonical_reordering_map_for_frag(h, h_junction_count, e_index, e):
    n_ori_atom = h.shape[0]
    if np.sum(h_junction_count) > 0:
        h_, e_index_, e_ = add_atom_to_junction(h, h_junction_count, e_index, e)
    else:
        h_, e_index_, e_ = h, e_index, e
    remolstar = reconstruct_to_rdmol(h_, e_index_, e_, remove_h=False)
    remol = reconstruct_to_rdmol(h_, e_index_, e_, remove_h=True)
    smi = Chem.MolToSmiles(remol)
    canmol = Chem.MolFromSmiles(smi)
    canmolh = Chem.AddHs(canmol)
    # get match
    match = np.array(remolstar.GetSubstructMatch(canmolh))
    rev_match = np.zeros_like(match)
    for i, v in enumerate(match):
        rev_match[v] = i
    # check match except for (*) atoms
    new_match = rev_match[:n_ori_atom]
    new_match = np.argsort(new_match)  # rank of h
    return smi, new_match


def get_canonical_reordering_map_for_frag_without_h(h, h_junction_count, e_index, e):
    n_ori_atom = h.shape[0]
    if np.sum(h_junction_count) > 0:
        h_, e_index_, e_ = add_atom_to_junction(h, h_junction_count, e_index, e)
    else:
        h_, e_index_, e_ = h, e_index, e
    remolstar = reconstruct_to_rdmol(h_, e_index_, e_, remove_h=True)
    remol = reconstruct_to_rdmol(h_, e_index_, e_, remove_h=True)
    smi = Chem.MolToSmiles(remol)
    canmol = Chem.MolFromSmiles(smi)
    canmolh = Chem.AddHs(canmol)
    # get match
    match = np.array(remolstar.GetSubstructMatch(canmolh))
    rev_match = np.zeros_like(match)
    for i, v in enumerate(match):
        rev_match[v] = i
    new_match = rev_match[:n_ori_atom]
    new_match = np.argsort(new_match)  # rank of h
    return smi, new_match


def get_canonical_reordering_map_for_frag_relaxed(h, h_junction_count, e_index, e):
    n_ori_atom = h.shape[0]
    if np.sum(h_junction_count) > 0:
        h_, e_index_, e_ = add_atom_to_junction(h, h_junction_count, e_index, e)
    else:
        h_, e_index_, e_ = h, e_index, e
    remolstar = reconstruct_to_rdmol(h_, e_index_, e_, remove_h=False, is_relaxed=True)
    # print(Chem.MolToSmiles(remolstar))
    remol = reconstruct_to_rdmol(h_, e_index_, e_, remove_h=True, is_relaxed=True)
    smi = Chem.MolToSmiles(remol)
    canmol = Chem.MolFromSmiles(smi)
    canmolh = Chem.AddHs(canmol)  # this will be in the fragment bag
    # get match
    match = np.array(remolstar.GetSubstructMatch(canmolh))
    rev_match = np.zeros_like(match)
    for i, v in enumerate(match):
        rev_match[v] = i
    # check match except for (*) atoms
    new_match = rev_match[:n_ori_atom]
    new_match = np.argsort(new_match)  # rank of h

    _rev_match = np.array(canmolh.GetSubstructMatch(canmolh))
    _match = np.zeros_like(_rev_match)
    for i, v in enumerate(_rev_match):
        _match[v] = i
    _new_match = _match[_match < n_ori_atom]
    # print(_new_match)
    # print(new_match)

    # print(Chem.MolToSmiles(canmolh))
    # print(Chem.MolToSmiles(remolstar))
    if len(new_match) == 0:
        # print(new_match)
        _rev_match = np.array(canmolh.GetSubstructMatch(canmolh))
        _match = np.zeros_like(_rev_match)
        for i, v in enumerate(_rev_match):
            _match[v] = i
        _new_match = _match[_match < n_ori_atom]
        # _new_match = np.argsort(_new_match)  # rank of h
        # print(_new_match)
        assert len(_new_match) == n_ori_atom, f"length mismatch"
        return smi, _new_match
    else:
        return smi, new_match


def _filter_all_valid_pairs(pairs, groups):
    def in_same_group(x, y, groups):
        for grp in groups:
            if x in grp and y in grp:
                return True
        return False

    num_cols = pairs.shape[1]
    valid_cols = []
    for i in range(num_cols):
        u = pairs[0, i]
        v = pairs[1, i]
        if not in_same_group(u, v, groups):
            valid_cols.append(i)
    valid_cols = np.array(valid_cols)
    if len(valid_cols) == 0:
        return np.array([], dtype=pairs.dtype)
    return valid_cols
    # return pairs[:, valid_cols]
