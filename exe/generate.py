import os
import shutil
import sys
from datetime import datetime
from pprint import pprint
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from rdkit import RDLogger

from fragfm.mol_generator import FragFMGenerator
from fragfm.utils.file import read_yaml_as_easydict

RDLogger.DisableLog("rdApp.*")


if __name__ == "__main__":
    cfg_fn = sys.argv[1]
    cfg = read_yaml_as_easydict(cfg_fn)

    # saving dirn
    cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
    cfg.save_dirn = f"save/generate/{cur_time}_{cfg.tag}"
    cfg.save_fn = f"save/generate/{cur_time}_{cfg.tag}/sample.txt"
    if "force_save_dirn" in cfg:
        cfg.save_dirn = cfg.force_save_dirn
        cfg.save_fn = os.path.join(cfg.force_save_dirn, "sample.txt")
    os.makedirs(cfg.save_dirn, exist_ok=True)
    print("Generation saved to:", cfg.save_dirn)
    shutil.copy(cfg_fn, os.path.join(cfg.save_dirn, "cfg.yaml"))

    # sampling
    sampler = FragFMGenerator(cfg)
    sampler.set_seed(cfg.seed)

    print("Start sampling...\n")
    for sample_idx in range(cfg.n_sample // cfg.bs + 1):
        print(f"Start generation {sample_idx + 1} / {cfg.n_sample // cfg.bs + 1}")
        x = sampler.sample_molecule_graph_dynamic()
        sampler.store_smis_from_coarse_graph(*x)
        print("Save SMILES...")
        sampler.save_smis(cfg.save_fn)

        # i want to print it 4 digits under
        print(f"Current generated: {len(sampler.gen_smis)}")
        print(
            f"Current validity: {100 * len(sampler.valid_smis) / len(sampler.gen_smis):.2f}"
        )
        print(
            f"Current uniqueness: {100 * len(sampler.unique_smis) / len(sampler.valid_smis):.2f}"
        )
        if sampler.is_disc_model:
            if sampler.prop_vals[0] == False:
                pass
            else:
                print(
                    f"Current condition MAE: {np.abs(np.array(sampler.prop_vals) - sampler.cfg.guide_val).mean():.4f}"
                )
        print()

    print(f"Generated samples saved:")
    print(cfg.save_dirn)
