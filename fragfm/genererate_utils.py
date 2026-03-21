import os
import random
import shutil
import sys
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
from easydict import EasyDict
from rdkit import Chem, RDLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch_geometric.utils import scatter
from tqdm import tqdm

from fragfm.utils.mat_ops import max_weight_matching_mask


def create_adj_from_coarse_e_index_and_frag_batch(a, b):
    """
    a: coarse_e_index (e.g. )
    b: batch of fragments
    """
    N = len(b)
    device = a.device
    adj_matrix = torch.zeros((N, N), device=device, dtype=torch.int)
    for i in range(a.shape[1]):
        source_val = a[0, i].item()
        target_val = a[1, i].item()
        source_indices = (b == source_val).nonzero(as_tuple=True)[0]
        target_indices = (b == target_val).nonzero(as_tuple=True)[0]
        for src in source_indices:
            for tgt in target_indices:
                adj_matrix[src, tgt] = 1
                adj_matrix[tgt, src] = 1  # ensure directed
    return adj_matrix


def adjacency_to_edge_index_triu(adj_matrix):
    adj_triu = torch.triu(adj_matrix, diagonal=1)
    edge_indices = adj_triu.nonzero(as_tuple=True)
    edge_index = torch.stack(edge_indices, dim=0)
    return edge_index


def coarse_graph_to_fine_graph_predictions(
    ae_model, frag_graphs, coarse_e_index, coarse_e, coarse_batch, z
):
    """
    frag_graphs: n_node graphs batched
    coares_e_index: [2, n_edge]
    coarse_e: [n_edge,]
    coarse_batch: [n_node]
    z: [bs]
    """
    frag_graphs.to("cuda")
    fine_h = frag_graphs.h
    fine_h_juncion_count = frag_graphs.h_junction_count
    fine_decomp_e_index = frag_graphs.e_index
    fine_decomp_e = frag_graphs.e
    fine_batch = coarse_batch[frag_graphs.batch]
    smis = frag_graphs.smi
    device = fine_h.device

    # get aux frag label (resolves duplicated fragments)
    fine_h_aux_label = []
    for i in range(coarse_batch.max().cpu().item() + 1):
        batch_mask = coarse_batch == i
        is_batch_idxs = torch.where(batch_mask)[0].tolist()
        occ_count, output = {}, []
        for k in is_batch_idxs:
            item = smis[k]
            if item not in occ_count:
                occ_count[item] = 0
                output.append(0)
            else:
                occ_count[item] += 1
                output.append(occ_count[item])
        fine_h_aux_label += output
    fine_h_aux_label = torch.Tensor(fine_h_aux_label).long().to(device)
    fine_h_aux_label = fine_h_aux_label[frag_graphs.batch]

    # get in frag label (resolves symmetric graphs such as *C(=O)CC*)
    label_changes = torch.cat(
        (
            torch.tensor([True], dtype=torch.bool).to(device),
            frag_graphs.batch[1:] != frag_graphs.batch[:-1],
        )
    )
    group_starts = torch.where(label_changes)[0]
    group_ends = torch.cat(
        (group_starts[1:], torch.tensor([len(frag_graphs.batch)]).to(device))
    )
    group_sizes = group_ends - group_starts
    fine_h_in_frag_label = (
        torch.arange(len(frag_graphs.batch)).to(device)
        - torch.repeat_interleave(group_starts, group_sizes)
    ).to(device)

    # now, get fine-edges to predict ({frag connected} and {junction-junction})
    # get: fine juncion-junction edges
    adj_mask_1 = fine_h_juncion_count.unsqueeze(0) * fine_h_juncion_count.unsqueeze(1)
    # get: frag connected fine edges
    connects = coarse_e_index[:, coarse_e.bool()].to(device)
    adj_mask_2 = create_adj_from_coarse_e_index_and_frag_batch(
        connects, frag_graphs.batch.to(device)
    )
    adj_mask = (adj_mask_1 * adj_mask_2).bool().int()
    ae_to_pred_index = adjacency_to_edge_index_triu(adj_mask)

    #
    fine_h_aux_label = fine_h_aux_label.clamp(max=29)
    pred_junction_prob = ae_model.decode(
        z,
        fine_h,
        fine_h_juncion_count,
        fine_h_in_frag_label,
        fine_h_aux_label,
        fine_decomp_e_index,
        fine_decomp_e,
        ae_to_pred_index,
        fine_batch,
    )

    out = {}
    out["h"] = fine_h
    out["h_junction_count"] = fine_h_juncion_count
    out["h_in_frag_label"] = fine_h_in_frag_label
    out["decomp_e_index"] = fine_decomp_e_index
    out["decomp_e"] = fine_decomp_e
    out["ae_to_pred_index"] = ae_to_pred_index
    out["ae_to_pred_prob"] = pred_junction_prob
    out["batch"] = fine_batch
    out["frag_batch"] = frag_graphs.batch
    out = EasyDict(out)
    return out


def debatch_fine_graph_dict_to_list(fine_graph_dict):
    bs = fine_graph_dict["batch"].max() + 1
    n_cumsum = 0
    out_list = []
    for i in range(bs):
        out = {}

        # unbatch nodes
        out["h"] = fine_graph_dict.h[fine_graph_dict.batch == i]
        out["h_junction_count"] = fine_graph_dict.h_junction_count[
            fine_graph_dict.batch == i
        ]
        frag_batch = fine_graph_dict.frag_batch[fine_graph_dict.batch == i]
        out["frag_batch"] = frag_batch - frag_batch.min()

        # unbatch deomp edges
        decomp_e_batch = fine_graph_dict.batch[fine_graph_dict.decomp_e_index[0]]
        out["decomp_e_index"] = (
            fine_graph_dict.decomp_e_index[:, decomp_e_batch == i] - n_cumsum
        )
        out["decomp_e"] = fine_graph_dict.decomp_e[decomp_e_batch == i]

        # unbatch predicted edges
        ae_to_pred_batch = fine_graph_dict.batch[fine_graph_dict.ae_to_pred_index[0]]
        out["ae_to_pred_index"] = (
            fine_graph_dict.ae_to_pred_index[:, ae_to_pred_batch == i] - n_cumsum
        )
        out["ae_to_pred_prob"] = fine_graph_dict.ae_to_pred_prob[ae_to_pred_batch == i]

        n_cumsum += out["h"].size(0)
        out = EasyDict(out)
        out_list.append(out)

    return out_list


def realize_single_fine_graph_dict(fine_graph_dict):
    """
    this works only for DEBATCHED inputs
    conduct BLOSSOM algorithm
    """
    device = fine_graph_dict.h.device
    d = fine_graph_dict

    # generate [n_atom, n_atom] adjacency with prob scores
    prob_score_mat_ext = torch.zeros(d.h.size(0), d.h.size(0)).to(device)
    prob_score_mat_ext_mask = prob_score_mat_ext.clone().bool()
    prob_score_mat_ext_mask[d.ae_to_pred_index[0], d.ae_to_pred_index[1]] = True
    prob_score_mat_ext_mask[d.ae_to_pred_index[1], d.ae_to_pred_index[0]] = True
    prob_score_mat_ext[d.ae_to_pred_index[0], d.ae_to_pred_index[1]] = d.ae_to_pred_prob
    prob_score_mat_ext[d.ae_to_pred_index[1], d.ae_to_pred_index[0]] = d.ae_to_pred_prob

    # get [n_junction_atom, n_junction_atom] from the fully extended
    row_has_val = prob_score_mat_ext_mask.any(dim=1)
    prob_score_mat = prob_score_mat_ext[row_has_val][:, row_has_val]
    prob_score_mat_mask = prob_score_mat_ext_mask[row_has_val][:, row_has_val]
    prob_score_mat[~prob_score_mat_mask] = -999.0

    def expand_matrix_by_duplicates(M, duplicates):
        M_row_expanded = torch.repeat_interleave(M, repeats=duplicates, dim=0)
        M_expanded = torch.repeat_interleave(M_row_expanded, repeats=duplicates, dim=1)
        return M_expanded

    def contract_matrix_by_duplicates(M_expanded, duplicates):
        row_blocks = torch.split(M_expanded, duplicates.tolist(), dim=0)
        merged_rows = [block.sum(dim=0, keepdim=True) for block in row_blocks]
        M_row_merged = torch.cat(merged_rows, dim=0)
        col_blocks = torch.split(M_row_merged, duplicates.tolist(), dim=1)
        merged_cols = [block.sum(dim=1, keepdim=True) for block in col_blocks]
        M_merged = torch.cat(merged_cols, dim=1)
        return M_merged

    recon_h_junction_count_skewd = d.h_junction_count[d.h_junction_count != 0]
    prob_score_mat_expand = expand_matrix_by_duplicates(
        prob_score_mat, recon_h_junction_count_skewd
    )
    bls_adj_mask = max_weight_matching_mask(prob_score_mat_expand).long()

    # sanitize the data
    if bls_adj_mask.size(0) != 0:  # when blossom pred needed
        bls_adj_mask = contract_matrix_by_duplicates(
            bls_adj_mask, recon_h_junction_count_skewd
        ).to(device)

        # re-fill the original adj matrix with blossom output
        pred_ae_adj = torch.zeros(d.h.size(0), d.h.size(0)).to(device).long()
        row_has_val_idx = torch.where(row_has_val)[0]  # shape [k]
        pred_ae_adj[row_has_val_idx[:, None], row_has_val_idx[None, :]] = bls_adj_mask
        # realize probability
        sel_recon_ae_to_pred_e_type = pred_ae_adj[
            d.ae_to_pred_index[0], d.ae_to_pred_index[1]
        ]

        recon_e_index = torch.cat([d.decomp_e_index, d.ae_to_pred_index], dim=1)
        recon_e = torch.cat([d.decomp_e, sel_recon_ae_to_pred_e_type], dim=0)
        recon_e_index_ = recon_e_index[:, recon_e > 0]
        recon_e_ = recon_e[recon_e > 0]

        recon_h_np = d.h.cpu().numpy()
        recon_e_index_np = recon_e_index_.cpu().numpy()
        recon_e_np = recon_e_.cpu().numpy()
    else:  # when dermined as single
        recon_e_index = d.decomp_e_index
        recon_e = d.decomp_e
        recon_e_index_ = recon_e_index[:, recon_e > 0]
        recon_e_ = recon_e[recon_e > 0]

        recon_h_np = d.h.cpu().numpy()
        recon_e_index_np = recon_e_index_.cpu().numpy()
        recon_e_np = recon_e_.cpu().numpy()

    return recon_h_np, recon_e_index_np, recon_e_np


def compute_rate_matrix_mask(gen_x_type, pred_x1_prob, t, noise, omega=0.0):
    n_dim = pred_x1_prob.size(0)
    n_type = pred_x1_prob.size(1)  # including mask
    mask_idx = n_type - 1
    device = gen_x_type.device
    x_mask_onehot = F.one_hot(
        torch.ones(n_dim).long() * n_type - 1, num_classes=n_type
    ).to(device)
    is_mask = (gen_x_type == mask_idx).float()
    x_rate = is_mask.unsqueeze(1) * pred_x1_prob
    x_rate = x_rate * ((1 + noise * t) / (1 - t)) + pred_x1_prob * omega / (1 - t)
    x_remask = (1.0 - is_mask.unsqueeze(1)) * x_mask_onehot * noise
    x_rate += x_remask
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), 0.0)
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), -x_rate.sum(dim=-1, keepdim=True))
    return x_rate


def compute_rate_matrix_mask_(gen_x_type, pred_x1_prob, t, noise):
    n_dim, n_type = pred_x1_prob.size()
    mask_idx = n_type - 1
    device = gen_x_type.device
    is_mask = gen_x_type == mask_idx  # [n_dim] → bool
    x_mask_onehot = F.one_hot(
        torch.full((n_dim,), mask_idx, device=device), num_classes=n_type
    ).float()
    x_rate = torch.zeros_like(pred_x1_prob)
    if is_mask.any():
        masked_idx = is_mask.nonzero(as_tuple=True)[0]  # indices where is_mask == True
        # slice pred_x1_prob only for masked positions
        x_rate[masked_idx] = pred_x1_prob[masked_idx] * ((1 + noise * t) / (1 - t))
    x_remask = (~is_mask).unsqueeze(1).float() * x_mask_onehot * noise
    x_rate += x_remask
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), 0.0)
    row_sum = x_rate.sum(dim=-1, keepdim=True)
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), -row_sum)
    return x_rate


def compute_rate_matrix_uniform(gen_x_type, pred_x1_prob, t, noise):
    n_dim = pred_x1_prob.size(0)
    n_type = pred_x1_prob.size(1)  # including mask
    device = gen_x_type.device
    x1_prob_at_xt = torch.gather(pred_x1_prob, -1, gen_x_type.unsqueeze(-1))
    k = (1 + noise + noise * (n_type - 1) * t) / (1 - t)
    x_rate = k * pred_x1_prob + noise * x1_prob_at_xt
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), 0.0)
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), -x_rate.sum(dim=-1, keepdim=True))
    return x_rate


def compute_rate_matrix_absorb(gen_x_type, pred_x1_prob, t, noise):
    n_dim = pred_x1_prob.size(0)
    n_type = pred_x1_prob.size(1)  # including mask
    mask_idx = 0
    device = gen_x_type.device
    x_mask_onehot = F.one_hot(
        torch.zeros(n_dim).long() * n_type, num_classes=n_type
    ).to(device)
    is_mask = (gen_x_type == mask_idx).float()
    x_rate = is_mask.unsqueeze(1) * pred_x1_prob
    x_rate = x_rate * ((1 + noise * t) / (1 - t))
    x_remask = (1.0 - is_mask.unsqueeze(1)) * x_mask_onehot * noise
    x_rate += x_remask
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), 0.0)
    x_rate.scatter_(-1, gen_x_type.unsqueeze(-1), -x_rate.sum(dim=-1, keepdim=True))
    return x_rate
