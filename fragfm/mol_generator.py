import os
import random
import shutil
import sys
from datetime import datetime
from pprint import pprint
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict
from rdkit import Chem, RDLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch_geometric.utils import scatter
from tqdm import tqdm

from fragfm.dataset import *
from fragfm.distort_scheduler import DistortScheduler
from fragfm.genererate_utils import (
    coarse_graph_to_fine_graph_predictions,
    compute_rate_matrix_absorb,
    compute_rate_matrix_mask,
    compute_rate_matrix_uniform,
    debatch_fine_graph_dict_to_list,
    realize_single_fine_graph_dict,
)
from fragfm.model.ae import FragJunctionAE
from fragfm.model.flow import CoarseGraphPropagate, FragToVect
from fragfm.process import reconstruct_to_rdmol
from fragfm.utils.file import read_yaml_as_easydict
from fragfm.utils.mat_ops import sample_from_prob

RDLogger.DisableLog("rdApp.*")


class FragFMGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fm_cfg = read_yaml_as_easydict(os.path.join(cfg.fm_model_dirn, "cfg.yaml"))
        self.ae_cfg = read_yaml_as_easydict(
            os.path.join(self.fm_cfg.ae_model_dirn, "cfg.yaml")
        )
        self.fm_cfg.latent_z_dim = self.ae_cfg.latent_z_dim

        # load latent transform paramters if trained with
        latent_transform_fn = os.path.join(
            cfg.fm_model_dirn, "latent_transform_param.pkl"
        )
        with open(latent_transform_fn, "rb") as f:
            self.cfg.latent_transform_param = pickle.load(f)

        # get distortion scheduler
        self.node_distort_scheduler = DistortScheduler(
            self.fm_cfg.node_distort_schedule
        )
        self.edge_distort_scheduler = DistortScheduler(
            self.fm_cfg.edge_distort_schedule
        )
        self.cont_distort_scheduler = DistortScheduler(
            self.fm_cfg.latent_z_distort_schedule
        )

        # get fragment lmdb env
        self.frag_env = lmdb.open(
            cfg.frag_data_dirn,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=100000000,
        )
        self.n_all_frag = int(self.frag_env.stat()["entries"])  # exc. mask

        # get test set and loader
        self.test_set = FragFMDataset(
            lmdb_fn=self.cfg.data_dirn,
            frag_lmdb_fn=self.cfg.frag_data_dirn,
            frag_smi_to_idx_fn=self.cfg.frag_smi_to_idx_fn,
            data_split=self.cfg.fragment_bag,
            debug=self.cfg.debug,
        )
        self.test_sampler = RandomSampler(
            self.test_set,
            num_samples=self.cfg.bs * (self.cfg.n_sample // self.cfg.bs + 1),
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.cfg.bs,
            sampler=self.test_sampler,
            collate_fn=collate_frag_fm_dataset,
            num_workers=0,
        )

        # load ae
        self.ae_model = FragJunctionAE(self.ae_cfg)
        self.ae_model.load_state_dict(
            torch.load(
                f"{self.fm_cfg.ae_model_dirn}/model_best.pt",
                map_location="cpu",
            )
        )
        self.ae_model.cuda()
        self.ae_model.eval()

        # load frag embedder and coarse gnn
        frag_embedder_sd = torch.load(
            f"{self.cfg.fm_model_dirn}/frag_embedder_ema_{cfg.model_cut}.pt",
            map_location="cpu",
        )
        coarse_gnn_sd = torch.load(
            f"{self.cfg.fm_model_dirn}/coarse_propagate_ema_{cfg.model_cut}.pt",
            map_location="cpu",
        )

        # load frag embedder and coarse gnn
        self.frag_embedder = FragToVect(self.fm_cfg)
        self.coarse_gnn = CoarseGraphPropagate(self.fm_cfg)
        self.frag_embedder.load_state_dict(frag_embedder_sd)
        self.coarse_gnn.load_state_dict(coarse_gnn_sd)
        self.frag_embedder.cuda()
        self.coarse_gnn.cuda()
        self.frag_embedder.eval()
        self.coarse_gnn.eval()

        all_frag_idxs = torch.arange(self.n_all_frag)
        frag_z_list, frag_junction_count_list = [], []
        pred_frag_prop_list = []
        for i in tqdm(
            range(0, self.n_all_frag, 200),
            desc="Compute Fragment Embeddings",
            ncols=80,
        ):
            cur_frag_idxs = all_frag_idxs[i : i + 200]
            cur_frag_graphs = self.get_batched_frag_graph(cur_frag_idxs)
            cur_frag_graphs.to("cuda")
            with torch.no_grad():
                if self.fm_cfg.frag_fully_connected_graph:
                    frag_zs = self.frag_embedder(
                        cur_frag_graphs.h,
                        cur_frag_graphs.h_junction_count,
                        cur_frag_graphs.e_index,
                        cur_frag_graphs.e,
                        cur_frag_graphs.g,
                        cur_frag_graphs.batch,
                        cat_mask=False,
                    )
                else:
                    frag_zs = self.frag_embedder(
                        cur_frag_graphs.h,
                        cur_frag_graphs.h_junction_count,
                        cur_frag_graphs.full_e_index,
                        cur_frag_graphs.full_e,
                        cur_frag_graphs.g,
                        cur_frag_graphs.batch,
                        cat_mask=False,
                    )

            frag_z_list.append(frag_zs)
            frag_junction_count_list.append(cur_frag_graphs.junction_count)
            del cur_frag_idxs, cur_frag_graphs

        self.all_frag_z = torch.cat(frag_z_list, dim=0).detach()
        self.all_frag_junction_count = torch.cat(
            frag_junction_count_list, dim=0
        ).detach()
        del frag_z_list, frag_junction_count_list

        # store the occurance of fragments
        self._store_frag_occurance()

        # generation tracker
        self.gen_smis = []  # include "X"
        self.valid_smis, self.unique_smis, self.novel_smis = [], [], []

        # when disc_model_dirn not exists in cfg, assign None
        if not hasattr(cfg, "disc_model_dirn"):
            self.cfg.disc_model_dirn = False
        if self.cfg.disc_model_dirn is False:
            self.is_disc_model = False
        else:
            self.is_disc_model = True
            # print it red
            print("\033[91mDiscriminator model detected, loading...\033[0m")
            print(f"Condtioning value: {self.cfg.guide_val}")
            print(f"Condtioning strength: {self.cfg.graph_guide_strength}")
            from fragfm.model.disc import CoarseGraphReadout, FragToVectReadout

            # get disc_cfg
            self.disc_cfg = read_yaml_as_easydict(
                os.path.join(cfg.disc_model_dirn, "cfg.yaml")
            )
            self.disc_cfg.latent_z_dim = self.ae_cfg.latent_z_dim

            # load models for discriminator
            self.disc_cfg.node_prior, self.disc_cfg.edge_prior = (
                self.fm_cfg.node_prior,
                self.fm_cfg.edge_prior,
            )
            self.disc_frag_embedder = FragToVectReadout(self.disc_cfg)
            self.disc_coarse_gnn = CoarseGraphReadout(self.disc_cfg)

            # load models
            self.disc_frag_embedder.load_state_dict(
                torch.load(
                    f"{self.cfg.disc_model_dirn}/frag_embedder_ema_best.pt",
                    map_location="cpu",
                ),
                strict=False,
            )
            self.disc_frag_embedder.cuda()
            self.disc_frag_embedder.eval()
            self.disc_coarse_gnn.load_state_dict(
                torch.load(
                    f"{self.cfg.disc_model_dirn}/coarse_propagate_ema_best.pt",
                    map_location="cpu",
                ),
                strict=False,
            )
            self.disc_coarse_gnn.cuda()
            self.disc_coarse_gnn.eval()
            self.prop_vals = []  # tracker

            # process fragments first
            save_expected_prop_dirn = os.path.join(
                self.cfg.disc_model_dirn, "pred_frag_prop.pt"
            )
            try:
                self.pred_frag_prop = torch.load(save_expected_prop_dirn)
                print("Fragment property prediction loaded")
            except:
                all_frag_idxs = torch.arange(self.n_all_frag)
                pred_frag_prop_list = []
                for i in tqdm(
                    range(0, self.n_all_frag, 256),
                    desc="Expected Fagment Property",
                    ncols=80,
                ):
                    cur_frag_idxs = all_frag_idxs[i : i + 256]
                    cur_frag_graphs = self.get_batched_frag_graph(cur_frag_idxs)
                    cur_frag_graphs.to("cuda")
                    with torch.no_grad() and torch.inference_mode():
                        if self.disc_cfg.frag_fully_connected_graph:
                            frag_zs, pred_frag_prop = self.disc_frag_embedder(
                                cur_frag_graphs.h,
                                cur_frag_graphs.h_junction_count,
                                cur_frag_graphs.e_index,
                                cur_frag_graphs.e,
                                cur_frag_graphs.g,
                                cur_frag_graphs.batch,
                            )
                        else:
                            frag_zs, pred_frag_prop = self.disc_frag_embedder(
                                cur_frag_graphs.h,
                                cur_frag_graphs.h_junction_count,
                                cur_frag_graphs.full_e_index,
                                cur_frag_graphs.full_e,
                                cur_frag_graphs.g,
                                cur_frag_graphs.batch,
                            )
                    pred_frag_prop_list.append(pred_frag_prop.cpu())
                    del cur_frag_idxs, cur_frag_graphs
                self.pred_frag_prop = (
                    torch.cat(pred_frag_prop_list, dim=0).detach().cpu()
                )
                torch.save(self.pred_frag_prop, save_expected_prop_dirn)
                print("Fragment property prediction computed")

    def set_seed(self, seed):
        print("Set seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_batched_frag_graph(self, frag_idxs):
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

    def get_n_frags(self):
        for batch in self.test_loader:
            _, graph = batch
            ns = []
            for i in range(graph.batch.max() + 1):
                n = (graph.batch == i).sum()
                ns.append(n.cpu().item())
            break
        return ns  # list

    def sample_molecule_graph_dynamic(self, n_frags=None):
        # when node prior is mask only
        if self.fm_cfg.node_prior not in ["mask"]:
            print("Dynamic sampling is only available when node prior is mask.")
            raise NotImplementedError

        if n_frags is None:
            for batch in self.test_loader:
                _, graph = batch
                ns = []
                for i in range(graph.batch.max() + 1):
                    n = (graph.batch == i).sum()
                    ns.append(n)
                break
            graph = self._make_dummy_graph_batch(ns)
        else:
            graph = self._make_dummy_graph_batch(n_frags)
        graph.to("cuda")
        device = graph.batch.device

        # get params
        n_node = graph.h.size(0)
        n_edge = graph.full_e.size(0)
        bs = graph.batch.max() + 1

        # get intial fragment subbag (for prior distribution)
        base_frag_idxs = self.random_select_frags_by_occurance(
            self.fm_cfg.n_base_frag,
            split=self.cfg.fragment_bag,
            temperature=self.cfg.frag_select_temperature,
        )
        cur_frag_idxs = base_frag_idxs
        cur_frag_zs = self.all_frag_z[cur_frag_idxs].to(device)
        cur_frag_junction_count = self.all_frag_junction_count[cur_frag_idxs].to(device)
        # add masks to end
        if self.fm_cfg.node_prior == "mask":
            cur_frag_idxs = torch.cat(
                [cur_frag_idxs, torch.Tensor([self.n_all_frag]).long()], dim=0
            )
            cur_frag_zs = torch.cat(
                [cur_frag_zs, self.frag_embedder.mask_type_frag_z], dim=0
            )
            cur_frag_junction_count = torch.cat(
                [cur_frag_junction_count, torch.zeros(1).to(device)], dim=0
            )
        else:
            raise NotImplementedError

        # node prior
        if self.frag_embedder.cfg.node_prior == "mask":
            glob_gen_h_type = torch.ones(n_node).to(device).long() * self.n_all_frag
        else:
            raise NotImplementedError
        # edge prior
        if self.frag_embedder.cfg.edge_prior == "mask":
            gen_e_type = torch.ones(n_edge).to(device).long() * 2
        elif self.frag_embedder.cfg.edge_prior == "absorb":
            gen_e_type = torch.zeros(n_edge).to(device).long()
        elif self.frag_embedder.cfg.edge_prior == "uniform":
            e_prior_prob = torch.ones(n_edge, 2).to(device) / 2
            gen_e_type = sample_from_prob(e_prior_prob)
        else:
            raise NotImplementedError
        # z prior
        gen_z = torch.randn(bs, self.ae_cfg.latent_z_dim).to(device)

        # conduct Euler steps
        d_pre_t_ = (self.cfg.t_max - self.cfg.t_min) / self.cfg.n_euler_step
        for step_idx, pre_t_ in enumerate(
            tqdm(
                torch.linspace(self.cfg.t_min, self.cfg.t_max, self.cfg.n_euler_step),
                desc="Euler steps",
                ncols=80,
            )
        ):
            # sample fragment bag for Euler step prediction
            exst_frag_idxs = torch.unique(glob_gen_h_type)  # inc. mask
            base_frag_idxs = self.random_select_frags_by_occurance(
                self.fm_cfg.n_base_frag,
                split=self.cfg.fragment_bag,
                temperature=self.cfg.frag_select_temperature,
            ).to(device)
            cur_frag_idxs = torch.cat(
                [
                    exst_frag_idxs,
                    base_frag_idxs,
                    torch.Tensor([self.n_all_frag]).long().to(device),
                ],
                dim=0,
            )  # inc. mask
            cur_frag_idxs = torch.unique(cur_frag_idxs)  # inc. mask
            n_cur_frag = cur_frag_idxs.size(0) - 1  # exc. mask
            cur_frag_zs = self.all_frag_z.to(device)[cur_frag_idxs[:-1]]  # exc. mask
            cur_frag_junction_count = self.all_frag_junction_count.to(device)[
                cur_frag_idxs[:-1]
            ].to(device)  # exc. mask

            # make prediction mask, union of {base frag} and {current state}
            frag_mask = torch.zeros([n_node, n_cur_frag + 1]).to(device)  # inc. M
            is_base_frag_mask = torch.isin(cur_frag_idxs, base_frag_idxs)  # inc. M
            frag_mask[:, is_base_frag_mask] = 1  # inc. M
            temp_cur_gen_h_type = torch.searchsorted(cur_frag_idxs, glob_gen_h_type)
            temp_cur_gen_h_onehot = F.one_hot(
                temp_cur_gen_h_type, num_classes=n_cur_frag + 1
            ).float()
            temp_cur_h_in_batch = scatter(
                temp_cur_gen_h_onehot, graph.batch, reduce="sum", dim=0
            )
            temp_cur_h_in_batch = temp_cur_h_in_batch[
                graph.batch
            ].bool()  # [n_node], inc. M
            frag_mask = frag_mask.bool() | temp_cur_h_in_batch  # 1.unsqueeze(1)
            frag_mask = frag_mask[:, :-1].bool()  # exc. M
            # add masks to end
            if self.fm_cfg.node_prior == "mask":
                cur_frag_zs = torch.cat(
                    [cur_frag_zs, self.frag_embedder.mask_type_frag_z.to(device)], dim=0
                )
                cur_frag_junction_count = torch.cat(
                    [cur_frag_junction_count, torch.zeros(1).to(device).to(device)],
                    dim=0,
                )
            else:
                raise NotImplementedError

            if step_idx == self.cfg.n_euler_step - 1:
                is_last = True
            else:
                is_last = False

            glob_gen_h_type, gen_e_type, gen_z = self._calc_euler_step(
                glob_gen_h_type,
                graph.full_e_index,
                gen_e_type,
                gen_z,
                graph.batch,
                cur_frag_idxs,
                cur_frag_zs,
                cur_frag_junction_count,
                pre_t_,
                d_pre_t_,
                is_last,
                frag_mask,
            )

        return (
            glob_gen_h_type,
            graph.full_e_index,
            gen_e_type,
            gen_z,
            graph.batch,
        )

    def store_smis_from_coarse_graph(
        self,
        gen_coarse_h,
        gen_coarse_e_index,
        gen_coarse_e,
        gen_z,
        gen_coarse_batch,
        # is_coarse_val,
    ):
        cur_gen_smis = []
        # revert latent z scaling
        if self.fm_cfg.latent_transform == "min_max":
            min_z, max_z = (
                self.cfg.latent_transform_param["min"].to(gen_z.device),
                self.cfg.latent_transform_param["max"].to(gen_z.device),
            )
            gen_z = (gen_z + 1) * 0.5 * (max_z - min_z) + min_z
        elif self.fm_cfg.latent_transform == "leave":
            pass
        else:
            raise NotImplementedError

        # get frag graphs
        gen_frag_graphs = self.get_batched_frag_graph(gen_coarse_h)

        # reconstruct to fine graph
        print("Fine graph generation from coarse graph and splitting...")
        fine_graph_dict = coarse_graph_to_fine_graph_predictions(
            self.ae_model,
            gen_frag_graphs,
            gen_coarse_e_index,
            gen_coarse_e,
            gen_coarse_batch,
            gen_z,
        )

        # debatch
        graph_dict_list = debatch_fine_graph_dict_to_list(fine_graph_dict)

        # conduct BLOSSOM algorithm (no batching, single)
        for i, graph_dict in enumerate(
            tqdm(graph_dict_list, desc="Molecule Reconstruction", ncols=80)
        ):
            # with the graph dict, prepare BLOSSOM input
            try:
                h, e_index, e = realize_single_fine_graph_dict(graph_dict)
            except:
                self.gen_smis.append("X")
                cur_gen_smis.append("X")
                continue

            # recon
            try:
                if "moses" in self.cfg.data_dirn:
                    m = reconstruct_to_rdmol(
                        h, e_index, e, is_relaxed=False, get_largest=True
                    )
                elif "guacamol" in self.cfg.data_dirn:
                    m = reconstruct_to_rdmol(
                        h, e_index, e, is_relaxed=True, get_largest=True
                    )
                elif "zinc250k" in self.cfg.data_dirn:
                    m = reconstruct_to_rdmol(
                        h, e_index, e, is_relaxed=True, get_largest=True
                    )
                elif "npgen" in self.cfg.data_dirn:
                    m = reconstruct_to_rdmol(
                        h, e_index, e, is_relaxed=False, get_largest=True
                    )
                else:
                    raise NotImplementedError
                smi = Chem.MolToSmiles(m)
                assert not "." in smi

                self.gen_smis.append(smi)
                cur_gen_smis.append(smi)
                self.valid_smis.append(smi)
                if not smi in self.unique_smis:
                    self.unique_smis.append(smi)

                # compute property
                if self.is_disc_model:
                    if "logp" in str(self.cfg.disc_model_dirn) + self.cfg.fm_model_dirn:
                        logp = Chem.Crippen.MolLogP(m)
                        self.prop_vals.append(logp)
                    elif (
                        "nring"
                        in str(self.cfg.disc_model_dirn) + self.cfg.fm_model_dirn
                    ):
                        nring = len(Chem.GetSymmSSSR(m))
                        self.prop_vals.append(nring)
                    else:
                        self.prop_vals.append(False)

            except Exception as err:
                self.gen_smis.append("X")
                cur_gen_smis.append("X")

        return cur_gen_smis

    def random_select_frags_by_occurance(
        self,
        n: int,
        split: str,
        temperature: float = 1.0,
    ):
        """
        Randomly select n fragments based on occurance
        """
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

        if temperature != 1.0:
            ps = np.exp(np.log(ps) / temperature)
            ps = ps / np.sum(ps)

        if self.is_disc_model:
            pred_frag_prop_mse = (self.pred_frag_prop[:, 0] - self.cfg.guide_val) ** 2
            bag_ratio = torch.exp(-self.cfg.bag_guide_strength * pred_frag_prop_mse)
            ps = ps * bag_ratio.cpu().numpy()
            ps = ps / np.sum(ps)

        ks = np.random.choice(list(range(self.n_all_frag)), size=n, replace=False, p=ps)
        return torch.sort(torch.Tensor(ks).long())[0]

    def _calc_prop_prediction_full(
        self,
        gen_h_onehot,
        full_e_index,
        gen_e_onehot,
        gen_z,
        batch,
        cur_frag_idxs,
        pre_t_,
    ):
        device = gen_h_onehot.device

        # check
        if self.fm_cfg.node_prior == "mask":
            is_cat_mask = True
            cur_frag_idxs_womask = cur_frag_idxs[:-1]
        else:
            is_cat_mask = False
            cur_frag_idxs_womask = cur_frag_idxs

        # get frag graphs
        frag_graph = self.get_batched_frag_graph(cur_frag_idxs_womask)
        frag_graph.to("cuda")
        if self.disc_cfg.frag_fully_connected_graph:
            frag_zs, _ = self.disc_frag_embedder(
                frag_graph.h,
                frag_graph.h_junction_count,
                frag_graph.e_index,
                frag_graph.e,
                frag_graph.g,
                frag_graph.batch,
                cat_mask=is_cat_mask,
            )
        else:
            frag_zs, _ = self.disc_frag_embedder(
                frag_graph.h,
                frag_graph.h_junction_count,
                frag_graph.full_e_index,
                frag_graph.full_e,
                frag_graph.g,
                frag_graph.batch,
                cat_mask=is_cat_mask,
            )
        model_t = torch.ones(batch.max() + 1).to(device) * pre_t_

        pred_mol_prop = self.disc_coarse_gnn(
            gen_h_onehot.float(),
            full_e_index,
            gen_e_onehot.float(),
            gen_z,
            batch,
            model_t,
            frag_zs,
        )
        return pred_mol_prop

    def _calc_euler_step(
        self,
        glob_gen_h_type,
        full_e_index,
        gen_e_type,
        gen_z,
        batch,
        cur_frag_idxs,
        cur_frag_zs,
        cur_frag_junction_count,
        pre_t_,
        d_pre_t_,
        is_last,
        frag_mask=None,
    ):
        device = glob_gen_h_type.device
        if self.fm_cfg.node_prior == "mask":
            n_cur_frag = cur_frag_zs.size(0) - 1
        else:
            n_cur_frag = cur_frag_zs.size(0)

        # convert time to interpolating time
        node_t_ = self.node_distort_scheduler.convert_time(pre_t_)
        edge_t_ = self.edge_distort_scheduler.convert_time(pre_t_)
        cont_t_ = self.cont_distort_scheduler.convert_time(pre_t_)
        node_next_t_ = self.node_distort_scheduler.convert_time(pre_t_ + d_pre_t_)
        edge_next_t_ = self.edge_distort_scheduler.convert_time(pre_t_ + d_pre_t_)
        cont_next_t_ = self.cont_distort_scheduler.convert_time(pre_t_ + d_pre_t_)
        d_node_t_ = node_next_t_ - node_t_
        d_edge_t_ = edge_next_t_ - edge_t_
        d_cont_t_ = cont_next_t_ - cont_t_
        model_t = torch.ones(batch.max() + 1).to(device) * pre_t_

        # covert glob_gen_h_type to gen_h_type, MASK in glob maps to last
        gen_h_type = torch.searchsorted(cur_frag_idxs.to(device), glob_gen_h_type)

        # process valency feature
        if self.fm_cfg.use_frag_valency:
            gen_h_junction_count = cur_frag_junction_count.cpu()[
                gen_h_type.cpu()
            ].cuda()
            gen_h_current_degree = scatter(
                (gen_e_type == 1).float(),
                full_e_index[0],
                dim_size=batch.size(0),
                reduce="sum",
            ) + scatter(
                (gen_e_type == 1).float(),
                full_e_index[1],
                dim_size=batch.size(0),
                reduce="sum",
            )
            gen_h_valency = gen_h_current_degree - gen_h_junction_count
            gen_h_valency = gen_h_valency.long()
        else:
            gen_h_valency = None

        # node onehot
        if self.fm_cfg.node_prior == "mask":
            gen_h_onehot = F.one_hot(gen_h_type, num_classes=n_cur_frag + 1)
        else:
            gen_h_onehot = F.one_hot(gen_h_type, num_classes=n_cur_frag)
        # edge onehot
        if self.fm_cfg.edge_prior == "mask":
            gen_e_onehot = F.one_hot(gen_e_type, num_classes=3)
        else:
            gen_e_onehot = F.one_hot(gen_e_type, num_classes=2)

        # pass through coarse_gnn
        with torch.no_grad():
            pred_h_vect, pred_e_logit, pred_z = self.coarse_gnn(
                gen_h_onehot.float(),
                full_e_index,
                gen_e_onehot.float(),
                gen_z,
                batch,
                model_t,
                cur_frag_zs,
                gen_h_valency,
            )

        # get logits for node
        pred_h_vect_ = pred_h_vect.unsqueeze(1)
        cur_frag_zs_ = cur_frag_zs.unsqueeze(0)
        pred_h_logit = (pred_h_vect_ * cur_frag_zs_).sum(dim=2)
        if self.fm_cfg.node_prior == "mask":
            pred_h_logit[:, n_cur_frag] = float("-inf")
        if frag_mask != None:  # mask non-base frags
            pred_h_logit[:, :n_cur_frag][~frag_mask] = float("-inf")

        if not hasattr(self.cfg, "frag_logit_temperature"):
            self.cfg.frag_logit_temperature = 1.0

        pred_h1_prob = torch.softmax(
            pred_h_logit / self.cfg.frag_logit_temperature, dim=1
        )
        pred_h1_prob = sample_from_prob(pred_h1_prob, return_onehot=True)

        # get edge
        if self.fm_cfg.edge_prior == "mask":
            pred_e_logit = torch.cat(
                [
                    pred_e_logit,
                    torch.ones_like(pred_e_logit[:, 0]).unsqueeze(1) * float("-inf"),
                ],
                dim=1,
            )
        pred_e1_prob = torch.softmax(pred_e_logit, dim=1)
        pred_e1_prob = sample_from_prob(pred_e1_prob, return_onehot=True)

        # last sampling is done by argmax (x1 prediction)
        if is_last:
            gen_h_type = torch.argmax(pred_h1_prob, dim=1)
            glob_gen_h_type = cur_frag_idxs.to(device)[gen_h_type]
            gen_e_type = torch.argmax(pred_e1_prob, dim=1)
            gen_z = pred_z
            return glob_gen_h_type, gen_e_type, gen_z

        # calc node rate
        if self.fm_cfg.node_prior == "mask":
            h_rate = compute_rate_matrix_mask(
                gen_h_type,
                pred_h1_prob,
                node_t_,
                self.cfg.node_noise,
            )
        elif self.fm_cfg.edge_prior == "uniform":
            h_rate = compute_rate_matrix_uniform(
                gen_h_type, pred_h1_prob, node_t_, self.cfg.node_noise
            )
        else:
            raise NotImplementedError
        # calc edge rate
        if self.fm_cfg.edge_prior == "mask":
            e_rate = compute_rate_matrix_mask(
                gen_e_type,
                pred_e1_prob,
                edge_t_,
                self.cfg.edge_noise,
            )
        elif self.fm_cfg.edge_prior == "absorb":
            e_rate = compute_rate_matrix_absorb(
                gen_e_type, pred_e1_prob, edge_t_, self.cfg.edge_noise
            )
        elif self.fm_cfg.edge_prior == "uniform":
            e_rate = compute_rate_matrix_uniform(
                gen_e_type, pred_e1_prob, edge_t_, self.cfg.edge_noise
            )
        else:
            raise NotImplementedError
        # calc z rate
        z_rate = (pred_z - gen_z) / (1.0 - cont_t_)

        # node
        pred_step_h_prob = (h_rate * d_node_t_).clamp(min=0.0, max=1.0)
        pred_step_h_prob.scatter_(-1, gen_h_type.unsqueeze(-1), 0.0)
        pred_step_h_prob.scatter_(
            -1,
            gen_h_type.unsqueeze(-1),
            (1.0 - torch.sum(pred_step_h_prob, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        pred_step_h_prob.clamp(min=0.0, max=1.0)
        # edge
        pred_step_e_prob = (e_rate * d_edge_t_).clamp(min=0.0, max=1.0)
        pred_step_e_prob.scatter_(-1, gen_e_type.unsqueeze(-1), 0.0)
        pred_step_e_prob.scatter_(
            -1,
            gen_e_type.unsqueeze(-1),
            (1.0 - torch.sum(pred_step_e_prob, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        pred_step_e_prob.clamp(min=0.0, max=1.0)
        # Euler step z
        gen_z = gen_z + z_rate * d_cont_t_

        # compute digress style guidnace rates
        if self.is_disc_model:
            with torch.enable_grad():
                gen_h_onehot_ = gen_h_onehot.float().detach().requires_grad_(True)
                gen_e_onehot_ = gen_e_onehot.float().detach().requires_grad_(True)
                gen_z_ = gen_z.float().detach().requires_grad_(True)
                pred_prop = self._calc_prop_prediction_full(
                    gen_h_onehot_,
                    full_e_index,
                    gen_e_onehot_,
                    gen_z_,
                    batch,
                    cur_frag_idxs,
                    pre_t_,
                )
            pred_prop_mu = pred_prop[:, 0]
            pred_prop_mse = (pred_prop_mu - self.cfg.guide_val) ** 2

            pred_prop_mse = pred_prop_mse.sum().backward()
            gen_z_grad = gen_z_.grad
            gen_h_grad, gen_e_grad = gen_h_onehot_.grad, gen_e_onehot_.grad
            h_ratio = torch.softmax(-gen_h_grad * self.cfg.graph_guide_strength, dim=-1)
            e_ratio = torch.softmax(-gen_e_grad * self.cfg.graph_guide_strength, dim=-1)
            pred_step_h_prob = pred_step_h_prob * h_ratio
            pred_step_e_prob = pred_step_e_prob * e_ratio
            pred_step_h_prob[torch.sum(pred_step_h_prob, dim=-1) == 0] = 1e-7
            pred_step_e_prob[torch.sum(pred_step_e_prob, dim=-1) == 0] = 1e-7
            gen_z = gen_z - gen_z_grad * self.cfg.graph_guide_strength

        # sample
        gen_h_type = sample_from_prob(pred_step_h_prob)
        glob_gen_h_type = cur_frag_idxs.to(device)[gen_h_type]
        gen_e_type = sample_from_prob(pred_step_e_prob)

        return glob_gen_h_type, gen_e_type, gen_z

    def _store_frag_occurance(self):
        train_occ_list, vaild_occ_list, test_occ_list, smi_list = [], [], [], []
        occ_list = []
        with self.frag_env.begin() as txn:
            for i in range(self.n_all_frag):
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

    def _make_dummy_graph(self, n_frag: int):
        h = torch.zeros(n_frag, self.all_frag_z.size(1)).long()
        e_index, e = [], []
        for i in range(n_frag):
            for j in range(n_frag):
                if i < j:
                    e_index.append([i, j])
                    e.append(1)
        e_index = torch.Tensor(e_index).long().t()
        e = torch.Tensor(e).long()
        graph = Data(h=h, num_nodes=n_frag, full_e_index=e_index, full_e=e)
        return graph

    def _make_dummy_graph_batch(self, n_frag_lst: list[int]):
        graphs = []
        for n_frag in n_frag_lst:
            graphs.append(self._make_dummy_graph(n_frag))
        return Batch.from_data_list(graphs)

    def save_smis(self, save_fn):
        with open(save_fn, "w") as f:
            for smi in self.gen_smis:
                f.write(f"{smi}\n")


if __name__ == "__main__":
    pass
