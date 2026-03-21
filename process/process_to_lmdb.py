import os
import random
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Draw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


from fragfm.process import create_lmdb_dataset, process_data, process_sample
from fragfm.utils.file import (
    read_coconut_fn_to_smiles_list,
    read_guacamol_smiles_fn_to_smiles_list,
    read_moses_fn_to_smiles_list,
)


def split_to_sublists(input_list, n):
    k, m = divmod(len(input_list), n)
    return [
        input_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    ]


if __name__ == "__main__":
    # print(f"[LOG] Start processing")
    data_type = sys.argv[1]  # moses, guacamol, ...
    decomp_method = sys.argv[2]  # brics, rbrics
    data_tag = sys.argv[3]  # data tag
    chunck_idx = int(sys.argv[4])  # integer
    n_chunk = int(sys.argv[5])  # integer
    is_debug = False

    if len(sys.argv) > 6:
        print("[ERROR] The script should be called with 5 arguments")
        print(
            "[ERROR] python process_to_lmdb.py data_type decomp_method data_tag chunck_idx n_chunk"
        )

    save_dirn = f"./data/processed/{data_type}_{decomp_method}_{data_tag}/"
    lmdb_dirn = f"./data/processed/{data_type}_{decomp_method}_{data_tag}/{chunck_idx}_{n_chunk}.lmdb"

    os.makedirs(save_dirn, exist_ok=True)

    if data_type == "moses":
        # files
        train_fn = "./data/raw/moses/train.csv"
        valid_fn = "./data/raw/moses/test.csv"
        test_fn = "./data/raw/moses/test_scaffolds.csv"
        # read smi and make
        train_smis = read_moses_fn_to_smiles_list(train_fn)
        valid_smis = read_moses_fn_to_smiles_list(valid_fn)
        test_smis = read_moses_fn_to_smiles_list(test_fn)
    elif data_type == "guacamol":
        # files
        train_fn = "./data/raw/guacamol/guacamol_v1_train.smiles"
        valid_fn = "./data/raw/guacamol/guacamol_v1_valid.smiles"
        test_fn = "./data/raw/guacamol/guacamol_v1_test.smiles"
        # read smi and make
        train_smis = read_guacamol_smiles_fn_to_smiles_list(train_fn)
        valid_smis = read_guacamol_smiles_fn_to_smiles_list(valid_fn)
        test_smis = read_guacamol_smiles_fn_to_smiles_list(test_fn)
    elif data_type == "npgen":
        train_fn = "./data/raw/npgen/train.txt"
        valid_fn = "./data/raw/npgen/valid.txt"
        test_fn = "./data/raw/npgen/test.txt"
        # read smi and make
        train_smis = read_coconut_fn_to_smiles_list(train_fn)
        valid_smis = read_coconut_fn_to_smiles_list(valid_fn)
        test_smis = read_coconut_fn_to_smiles_list(test_fn)
    elif data_type == "zinc250k":
        train_fn = "./data/raw/zinc250k/250k_rndm_zinc_drugs_clean.smi"
        smis = read_coconut_fn_to_smiles_list(train_fn)
        train_smis = smis[:-15000]
        valid_smis = smis[-15000:-5000]
        test_smis = smis[-5000:]
    elif data_type == "debug":
        # files
        train_fn = "./data/raw/debug/train.csv"
        valid_fn = "./data/raw/debug/test.csv"
        test_fn = "./data/raw/debug/test_scaffolds.csv"
        # read smi and make
        train_smis = read_moses_fn_to_smiles_list(train_fn)
        valid_smis = read_moses_fn_to_smiles_list(valid_fn)
        test_smis = read_moses_fn_to_smiles_list(test_fn)

    else:
        raise NotImplementedError

    if is_debug:
        train_smis = train_smis[:10]
        valid_smis = valid_smis[:10]
        test_smis = test_smis[:10]

    train_data = [
        {
            "smi": smi,
            "key": f"train_{i}",
            "data_type": data_type,
            "decomp_method": decomp_method,
        }
        for i, smi in enumerate(train_smis)
    ]
    valid_data = [
        {
            "smi": smi,
            "key": f"valid_{i}",
            "data_type": data_type,
            "decomp_method": decomp_method,
        }
        for i, smi in enumerate(valid_smis)
    ]
    test_data = [
        {
            "smi": smi,
            "key": f"test_{i}",
            "data_type": data_type,
            "decomp_method": decomp_method,
        }
        for i, smi in enumerate(test_smis)
    ]

    all_data = train_data + valid_data + test_data
    all_data_chunks = split_to_sublists(all_data, n_chunk)
    to_process_data = all_data_chunks[chunck_idx]
    # print(f"[LOG] Number of data: {len(to_process_data)}", flush=True)
    """print("[LOG] First and last data point", flush=True)
    print(to_process_data[0], flush=True)
    print(to_process_data[-1],  flush=True)"""

    # print(f"[LOG] Start processing {len(to_process_data)} data", flush=True)
    data = process_data(to_process_data)
    print(f"[LOG] Processed {len(data)} data out of {len(to_process_data)}", flush=True)

    # print(f"[LOG] Start saving to: {lmdb_dirn}", flush=True)
    create_lmdb_dataset(data, lmdb_dirn, map_size=1e12)
    print(
        f"[LOG] Done processing molecules {data_type} {decomp_method} {data_tag} {chunck_idx} {n_chunk}",
        flush=True,
    )
