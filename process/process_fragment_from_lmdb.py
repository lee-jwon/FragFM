import logging
import os
import pickle
import random
import sys
import time

import lmdb
import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Draw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from fragfm.process import create_lmdb_dataset, process_frag
from fragfm.utils.graph_ops import *


def split_to_sublists(input_list, n):
    k, m = divmod(len(input_list), n)
    return [
        input_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    ]


if __name__ == "__main__":
    data_type = sys.argv[1]
    frag_method = sys.argv[2]
    data_tag = sys.argv[3]
    print(
        f"[LOG] Start processing fragments {data_type} {frag_method} {data_tag}",
        flush=True,
    )

    # lmdb_fn = f"data/processed/moses_241121.lmdb"
    lmdb_fn = f"data/processed/{data_type}_{frag_method}_{data_tag}.lmdb"

    env = lmdb.open(
        lmdb_fn,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        map_size=50000000000,
    )

    frag_smi_dict_train = {}
    frag_smi_dict_valid = {}
    frag_smi_dict_test = {}
    frag_smi_dict_all = {}

    with env.begin() as txn:
        cursor = txn.cursor()
        keys = [key for key, _ in cursor if "" in key.decode()]  # train + valid + test
        keys = keys  # [:1000]
        for key in tqdm(keys, disable=True):
            with env.begin() as txn:
                sample = pickle.loads(txn.get(key))
            for smi in sample["frag_smi_list"]:
                if b"train" in key:
                    if smi in frag_smi_dict_train:
                        frag_smi_dict_train[smi] += 1
                    else:
                        frag_smi_dict_train[smi] = 1
                elif b"valid" in key:
                    if smi in frag_smi_dict_valid:
                        frag_smi_dict_valid[smi] += 1
                    else:
                        frag_smi_dict_valid[smi] = 1
                elif b"test" in key:
                    if smi in frag_smi_dict_test:
                        frag_smi_dict_test[smi] += 1
                    else:
                        frag_smi_dict_test[smi] = 1
                if smi in frag_smi_dict_all:
                    frag_smi_dict_all[smi] += 1
                else:
                    frag_smi_dict_all[smi] = 1

    frag_data, frag_smiles_to_idx = [], {}
    for i, (k, v) in enumerate(
        tqdm(
            sorted(frag_smi_dict_all.items(), key=lambda item: item[1], reverse=True),
            disable=False,
        )
    ):
        sample = {}
        sample["key"] = f"frag_{i}"
        sample["smi"] = k
        sample["occurance"] = v
        sample["train_occurance"] = frag_smi_dict_train.get(k, 0)
        sample["valid_occurance"] = frag_smi_dict_valid.get(k, 0)
        sample["test_occurance"] = frag_smi_dict_test.get(k, 0)
        sample["occurance"] = v
        if data_type in ["guacamol", "zinc250k"]:
            processed_sample = process_frag(k, is_relax=True)
        elif data_type in ["debug", "moses", "npgen"]:
            processed_sample = process_frag(k, is_relax=False)
        else:
            raise NotImplementedError
        sample.update(processed_sample)
        frag_data.append(sample)
        frag_smiles_to_idx[k] = i

    # print(frag_smiles_to_idx)
    with open(
        f"./data/processed/{data_type}_{frag_method}_{data_tag}_fragment_to_idx.pkl",
        "wb",
    ) as f:
        pickle.dump(frag_smiles_to_idx, f)
    create_lmdb_dataset(
        frag_data,
        f"./data/processed/{data_type}_{frag_method}_{data_tag}_fragment.lmdb",
    )
    print(
        f"[LOG] Done processing fragments {data_type} {frag_method} {data_tag}",
        flush=True,
    )
