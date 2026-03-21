import os
import pickle
import random
from multiprocessing import Pool, cpu_count

import lmdb
import numpy as np
import parmap
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from fragfm.utils.graph_ops import sparse_edge_to_fully_connected_edge


def count_rings_from_smiles(smiles: str) -> int:
    """
    Given a SMILES string, return the number of rings in the molecule.

    :param smiles: SMILES representation of a molecule.
    :return: Number of rings in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    ring_info = mol.GetRingInfo()
    return ring_info.NumRings()


def convert_array_to_tensor_in_dict(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            data_dict[key] = torch.tensor(value, dtype=torch.as_tensor(value).dtype)
    return data_dict


def collate_frag_fm_dataset(sample_list):
    graph = Batch.from_data_list([x[0] for x in sample_list])
    coarse_graph = Batch.from_data_list([x[1] for x in sample_list])
    return graph, coarse_graph


class DiscDataset(Dataset):
    def __init__(
        self, lmdb_fn, frag_lmdb_fn, frag_smi_to_idx_fn, data_split="train", debug=False
    ):
        super().__init__()
        self.env = lmdb.open(
            lmdb_fn,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=50000000000,
        )
        self.frag_env = lmdb.open(
            frag_lmdb_fn,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=100000000,
        )

        self.keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            self.keys = [key for key, _ in cursor if data_split in key.decode()]

        if debug == "single":
            self.keys = [self.keys[1]] * 1000
            print("[WARNING] Debugging with single dataset")
        elif debug == "1K":
            self.keys = self.keys[:1000]
            print("[WARNING] Debugging with 1K dataset")
        elif debug == "10K":
            self.keys = self.keys[:10000]
            print("[WARNING] Debugging with 10K dataset")
        elif debug == False:
            pass
        else:
            raise NotImplementedError

        self.length = len(self.keys)

        # check fragment keys
        self.frag_keys = []
        with self.frag_env.begin() as txn:
            cursor = txn.cursor()
            self.frag_keys = [key for key, _ in cursor if "frag" in key.decode()]

        # get frag smi to idx dictionary
        with open(frag_smi_to_idx_fn, "rb") as f:
            self.frag_smi_to_idx = pickle.load(f)

        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def get_n_frag(self):
        return len(self.frag_smi_to_idx)

    def __getitem__(self, idx):
        # Access the filtered key by index
        key = self.keys[idx]
        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(key))
            full_coarse_e_index, full_coarse_e = sparse_edge_to_fully_connected_edge(
                sample["coarse_e_index"],
                np.ones_like(sample["coarse_e_index"][0, :]),
                n_node=len(sample["frag_smi_list"]),
                pad_val=0,
            )
            full_coarse_e_index = torch.Tensor(full_coarse_e_index).long()
            full_coarse_e = torch.Tensor(full_coarse_e).long()

            sample = convert_array_to_tensor_in_dict(sample)

            # get auxillary fragment label (index of fragment)
            occ_count = {}
            output = []
            for item in sample["frag_smi_list"]:
                if item not in occ_count:
                    occ_count[item] = 0
                    output.append(0)
                else:
                    occ_count[item] += 1
                    output.append(occ_count[item])
            h_aux_frag_label = torch.tensor(output)[sample["h_frag_batch"]]

            # get smis to index
            smi_idxs = []
            for smi in sample["frag_smi_list"]:
                smi_idxs.append(self.frag_smi_to_idx[smi])
            coarse_h = torch.Tensor(smi_idxs).long()

            # calculate logp
            mol = Chem.MolFromSmiles(sample["smi"])
            logp = Descriptors.MolLogP(mol)
            n_rings = count_rings_from_smiles(sample["smi"])

            # get graph
            graph = Data(
                key=sample["key"],
                smi=sample["smi"],
                smis=sample["frag_smi_list"],
                num_nodes=sample["h"].size(0),
                h=sample["h"].long(),
                h_junction_count=sample["h_junction_count"].long(),
                h_in_frag_label=sample["h_in_frag_label"].long(),
                h_aux_frag_label=h_aux_frag_label,
                h_frag_batch=sample["h_frag_batch"].long(),
                e_index=sample["e_index"].long(),
                e=sample["e"].long(),
                decomp_e_index=sample["decomp_e_index"],
                decomp_e=sample["decomp_e"],
                ae_to_pred_index=sample["ae_to_pred_index"],
                ae_to_pred=sample["ae_to_pred"],
                logp=torch.Tensor([logp]),
                nrgins=torch.Tensor([n_rings]),
            )
            # coarse graph
            coarse_graph = Data(
                key=sample["key"],
                smis=sample["frag_smi_list"],
                h=coarse_h,
                num_nodes=len(sample["frag_smi_list"]),
                e_index=sample["coarse_e_index"].long(),
                full_e_index=full_coarse_e_index.long(),
                full_e=full_coarse_e.long(),
            )

        return graph, coarse_graph

    def get_frags_from_index_tensor(self, frag_idxs):
        """t = self.frag_smi_to_idx["*C(=O)N=c1[nH]c2ccc(S(=O)CCC)cc2[nH]1"]
        print(t)"""
        graphs = []
        with self.frag_env.begin() as txn:
            for i in frag_idxs:
                key = f"frag_{int(i)}"  # Ensure `i` is converted to an integer if it's from a tensor
                data = txn.get(key.encode())  # Encode key as bytes
                sample = pickle.loads(data)

                # convert tensor to fully connected
                full_e_index, full_e = sparse_edge_to_fully_connected_edge(
                    sample["e_index"], sample["e"], sample["h"].shape[0]
                )

                sample = convert_array_to_tensor_in_dict(sample)
                graph = Data(
                    smi=sample["smi"],
                    h=sample["h"].long(),
                    h_junction_count=sample["h_junction_count"].long(),
                    num_nodes=sample["h"].size(0),
                    e_index=sample["e_index"].long(),
                    e=sample["e"].long(),
                    full_e_index=torch.Tensor(full_e_index).long(),
                    full_e=torch.Tensor(full_e).long(),
                    g=torch.cat(
                        [sample["g"], torch.Tensor([sample["h_junction_count"].sum()])],
                        dim=0,
                    )
                    .unsqueeze(0)
                    .float(),
                )
                graphs.append(graph)
        graph = Batch.from_data_list(graphs)
        return graph
