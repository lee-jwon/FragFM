import os
import pickle
import random
from multiprocessing import Pool, cpu_count

import lmdb
import numpy as np
import parmap
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from fragfm.utils.graph_ops import sparse_edge_to_fully_connected_edge


def convert_array_to_tensor_in_dict(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            data_dict[key] = torch.tensor(value, dtype=torch.as_tensor(value).dtype)
    return data_dict


def calculate_properties(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"logp": None, "nring": None, "tpsa": None, "qed": None}
        logp = Descriptors.MolLogP(mol) if mol else None
        num_rings = Descriptors.RingCount(mol) if mol else None
        tpsa = Descriptors.TPSA(mol) if mol else None
        qed = QED.qed(mol) if mol else None
        return {"logp": logp, "nring": num_rings, "tpsa": tpsa, "qed": qed}
    except Exception:
        # Return None for all values if an error occurs
        return {"logp": None, "nring": None, "tpsa": None, "qed": None}


class FragJunctionAEDataset(Dataset):
    def __init__(self, lmdb_path, data_split="train", debug=False):
        super().__init__()
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=50000000000,
        )

        self.keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            self.keys = [key for key, _ in cursor if data_split in key.decode()]

        if debug == "single":
            self.keys = [self.keys[0]]
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Access the filtered key by index
        key = self.keys[idx]
        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(key))
            sample = convert_array_to_tensor_in_dict(sample)

            # get auxillary fragment label (occurance idx count)
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

            # get graph
            graph = Data(
                key=sample["key"],
                smi=sample["smi"],
                smis=sample["frag_smi_list"],
                num_nodes=sample["h"].size(0),
                h=sample["h"],
                h_junction_count=sample["h_junction_count"],
                h_in_frag_label=sample["h_in_frag_label"],
                h_aux_frag_label=h_aux_frag_label,
                h_frag_batch=sample["h_frag_batch"],
                e_index=sample["e_index"],
                e=sample["e"],
                decomp_e_index=sample["decomp_e_index"],
                decomp_e=sample["decomp_e"],
                ae_to_pred_index=sample["ae_to_pred_index"],
                ae_to_pred=sample["ae_to_pred"],
            )
        return graph


def collate_frag_junction_ae_dataset(sample_list):
    b = Batch.from_data_list(sample_list)
    return b


class FragFMDataset(Dataset):
    def __init__(
        self,
        lmdb_fn,
        frag_lmdb_fn,
        frag_smi_to_idx_fn,
        data_split="train",
        debug=False,
        calc_prop=False,
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
        self.calc_prop = calc_prop

        # load calculated properties if calc_prop is "smina"
        if calc_prop:
            if "smina" in calc_prop:
                print("[WARNING] Loading smina properties")
                import csv

                if "fa7" in calc_prop:
                    print("[WARNING] Loading fa7 properties")
                    with open("data/raw/zinc250k_smina/fa7_score.csv", newline="") as f:
                        reader = csv.DictReader(f)
                        self.smiles_to_smina = {
                            row["smiles"]: float(row["score"]) for row in reader
                        }
                elif "jak2" in calc_prop:
                    print("[WARNING] Loading jak2 properties")
                    with open(
                        "data/raw/zinc250k_smina/jak2_score.csv", newline=""
                    ) as f:
                        reader = csv.DictReader(f)
                        self.smiles_to_smina = {
                            row["smiles"]: float(row["score"]) for row in reader
                        }
                elif "parp1" in calc_prop:
                    print("[WARNING] Loading parp1 properties")
                    with open(
                        "data/raw/zinc250k_smina/parp1_score.csv", newline=""
                    ) as f:
                        reader = csv.DictReader(f)
                        self.smiles_to_smina = {
                            row["smiles"]: float(row["score"]) for row in reader
                        }
                else:
                    raise NotImplementedError
            elif "npscore" in calc_prop:
                from rdkit.Contrib.NP_Score import npscorer

                self.np_scorer = npscorer.readNPModel("eval/publicnp.model.gz")

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
            self.n_total_frag = len(self.frag_keys)
            print(f"Total number of fragments are {len(self.frag_keys)}")

        # get frag smi to idx dictionary
        with open(frag_smi_to_idx_fn, "rb") as f:
            self.frag_smi_to_idx = pickle.load(f)

        self.length = len(self.keys)

        self._get_frag_occurance_list()

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

            # if self.calc_prop:
            if self.calc_prop:
                if "smina" in self.calc_prop:
                    smi = sample["smi"]
                    if smi in self.smiles_to_smina:
                        cond = self.smiles_to_smina[smi]  # scale
                    else:
                        cond = 999
                elif "npscore" in self.calc_prop:
                    from rdkit.Contrib.NP_Score import npscorer

                    smi = sample["smi"]
                    mol = Chem.MolFromSmiles(smi)
                    cond = npscorer.scoreMol(mol, self.np_scorer)
                else:
                    cond = calculate_properties(sample["smi"])[self.calc_prop]
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
                )
                # coarse graph
                coarse_graph = Data(
                    prop=torch.Tensor([cond]),
                    key=sample["key"],
                    smis=sample["frag_smi_list"],
                    h=coarse_h,
                    num_nodes=len(sample["frag_smi_list"]),
                    e_index=sample["coarse_e_index"].long(),
                    full_e_index=full_coarse_e_index.long(),
                    full_e=full_coarse_e.long(),
                )
                return graph, coarse_graph

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
                    junction_count=torch.Tensor([sample["h_junction_count"].sum()]),
                )
                graphs.append(graph)
        graph = Batch.from_data_list(graphs)
        return graph

    def _get_frag_occurance_list(self):
        train_occ_list, vaild_occ_list, test_occ_list, smi_list = [], [], [], []
        occ_list = []
        with self.frag_env.begin() as txn:
            for i in range(self.n_total_frag):
                key = f"frag_{int(i)}"
                data = txn.get(key.encode())
                sample = pickle.loads(data)
                train_occ_list.append(sample["train_occurance"])
                vaild_occ_list.append(sample["valid_occurance"])
                test_occ_list.append(sample["test_occurance"])
                occ_list.append(sample["occurance"])
                smi_list.append(sample["smi"])
        self.train_occ_list, self.valid_occ_list, self.test_occ_list = (
            train_occ_list,
            vaild_occ_list,
            test_occ_list,
        )
        self.occ_list = occ_list
        self.smi_list = smi_list

    def random_select_frags_by_occurance(
        self, n, split, temperature=1.0, get_prob=False
    ):
        if split == "train":
            occ_list = self.train_occ_list
        elif split == "valid":
            occ_list = self.valid_occ_list
        elif split == "test":
            occ_list = self.test_occ_list
        elif split == "all":
            occ_list = self.occ_list
        else:
            raise NotImplementedError
        ps = np.array(occ_list)
        ps = ps / np.sum(ps)

        # apply temperature
        if temperature != 1.0:
            ps = np.exp(np.log(ps) / temperature)
            ps = ps / np.sum(ps)

        if True:
            ks = np.random.choice(
                list(range(self.n_total_frag)), size=n, replace=False, p=ps
            )

        if get_prob:
            sorted_idx = np.argsort(ks)
            return torch.Tensor(ks).long()[sorted_idx], torch.Tensor(ps[ks])[sorted_idx]
        else:
            return torch.Tensor(ks).long()

    def get_n_frag(self):
        return len(self.occ_list)


def collate_frag_fm_dataset(sample_list):
    graph = Batch.from_data_list([x[0] for x in sample_list])
    coarse_graph = Batch.from_data_list([x[1] for x in sample_list])
    return graph, coarse_graph
