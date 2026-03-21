import os  # #
import random
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Draw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from fragfm.dataset import *
from fragfm.model.ae import FragJunctionAE as FragJunctionAE
from fragfm.process import *
from fragfm.utils.file import *
from fragfm.utils.graph_ops import *

RDLogger.DisableLog("rdApp.*")


def reconstruct_single_epoch(confs, model, data_loader):
    total_loss, total_recon_loss, total_reg_loss = 0.0, 0.0, 0.0
    n_edge, n_graph = 0, 0
    n_correct_edge, n_correct_graph = 0, 0
    n_batch = 0
    result = {}
    z_list = []
    for graph in tqdm(data_loader):
        graph.to("cuda")

        z, _ = model.encode(
            graph.h,
            graph.h_junction_count,
            graph.h_in_frag_label,
            graph.h_aux_frag_label,
            graph.e_index,
            graph.e,
            graph.batch,
        )
        # check robustness on z
        # z = z + torch.randn_like(z) * 0.2
        pred_jxn = model.decode(
            z,
            graph.h,
            graph.h_junction_count,
            graph.h_in_frag_label,
            graph.h_aux_frag_label,
            graph.decomp_e_index,
            graph.decomp_e,
            graph.ae_to_pred_index,
            graph.batch,
        )
        z_list.append(z.cpu().detach())

        pred_jxn = torch.sigmoid(pred_jxn)
        if False:
            pred_jxn = graph.ae_to_pred.float()

        recon_loss = F.binary_cross_entropy(
            pred_jxn, graph.ae_to_pred.float(), reduction="mean"
        )  # per

        # get accrucay
        pred_batch = graph.batch[graph.ae_to_pred_index[0]]
        wrong_mask = (
            (pred_jxn > 0.5) != graph.ae_to_pred
        ).int()  # 1 if wrong -> >1 if wrong, 0 if correct
        wrong_graph_mask = (scatter(wrong_mask, pred_batch, reduce="sum") != 0).int()
        n_edge += wrong_mask.size(0)
        n_graph += wrong_graph_mask.size(0)
        n_correct_edge += (1 - wrong_mask).sum().int().item()
        n_correct_graph += (1 - wrong_graph_mask).sum().int().item()

        total_recon_loss += recon_loss.item()
        n_batch += 1

        # make reconstruction
        n_cumsum = 0
        for i in range(graph.batch.max().item() + 1):
            ori_smi = graph.smi[i]
            decomp_batch = graph.batch[graph.decomp_e_index[0]]
            recon_e_index = graph.decomp_e_index[:, decomp_batch == i] - n_cumsum
            recon_e = graph.decomp_e[decomp_batch == i]

            recon_e_index_ = graph.ae_to_pred_index[:, pred_batch == i]
            # recon_e_ = graph.ae_to_pred[pred_batch == i]
            recon_e_ = (pred_jxn > 0.5).int()[pred_batch == i]
            recon_e_index_ = recon_e_index_[:, recon_e_ == 1] - n_cumsum
            recon_e_ = recon_e_[recon_e_ == 1]

            h = graph.h[graph.batch == i].cpu().numpy()
            e_index = torch.cat([recon_e_index, recon_e_index_], dim=1).cpu().numpy()
            e = torch.cat([recon_e, recon_e_], dim=0).cpu().numpy()
            n_cumsum += h.shape[0]

            try:
                mh = reconstruct_to_rdmol(h, e_index, e, remove_h=False)
                m = reconstruct_to_rdmol(h, e_index, e, remove_h=True)
                regen_smi = Chem.MolToSmiles(m)
                # assert regen_smi == ori_smi, regen_smi
            except Exception as e:
                # print(e)
                # print(graph.key[i])
                # print(ori_smi)
                # print(Chem.MolToSmiles(mh))
                # print()
                pass

    result["recon_loss"] = total_recon_loss / n_batch
    result["reg_loss"] = total_reg_loss / n_batch
    result["edge_accuracy"] = n_correct_edge / n_edge
    result["graph_accuracy"] = n_correct_graph / n_graph

    print("-" * 80)
    print(f"{'Number of molecules':<30}: {n_graph}")
    print(f"{'Reconstruction Loss':<30}: {result['recon_loss']:.6f}")
    print(f"{'Regularization Loss':<30}: {result['reg_loss']:.6f}")
    print(f"{'Edge Accuracy (%)':<30}: {result['edge_accuracy'] * 100:.2f}")
    print(f"{'Graph Accuracy (%)':<30}: {result['graph_accuracy'] * 100:.2f}")

    return result, z_list


if __name__ == "__main__":
    model_dirn = sys.argv[1]
    data_dirn = sys.argv[2]

    cfg_fn = os.path.join(model_dirn, "cfg.yaml")
    cfg = read_yaml_as_easydict(cfg_fn)
    cfg.data_dirn = data_dirn

    # cfg.data_dirn = "data/processed/coconut_filtered_brics_250123.lmdb"
    test_set = FragJunctionAEDataset(cfg.data_dirn, data_split="test", debug="10K")
    print(f"{len(test_set)}")

    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_frag_junction_ae_dataset,
        num_workers=4,
    )

    # model
    model = FragJunctionAE(cfg)
    sd = torch.load(os.path.join(model_dirn, "model_best.pt"))
    load_state_log = model.load_state_dict(sd)
    print(load_state_log)
    model.cuda()
    model.eval()

    with torch.no_grad():
        valid_result, z_list = reconstruct_single_epoch(None, model, test_loader)
    z = torch.cat(z_list, dim=0)
