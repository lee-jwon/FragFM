import copy
import math
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime

import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch_geometric.utils import scatter
from tqdm import tqdm

import wandb
from fragfm.dataset import *
from fragfm.distort_scheduler import DistortScheduler
from fragfm.model.ae import FragJunctionAE
from fragfm.model.disc import CoarseGraphReadout, FragToVectReadout
from fragfm.process import *
from fragfm.utils.file import *
from fragfm.utils.graph_ops import *


def sample_from_prob(x):
    n = x.size(1)
    x = x / x.sum(dim=1, keepdim=True)
    x = torch.multinomial(x, num_samples=1).squeeze(-1)
    return x


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k in msd.keys():
            if msd[k].dtype.is_floating_point:
                esd[k].data.mul_(decay).add_(msd[k].data, alpha=1 - decay)
            else:
                esd[k] = msd[k]  # int/bool 값은 그냥 복사


def kl_divergence_normal(x, eps=1e-8):
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False) + eps
    kl = 0.5 * (var + mean**2 - torch.log(var) - 1)
    return kl.mean()


def process_single_epoch(
    cfg,
    ae_model,
    frag_embedder,
    coarse_gnn,
    data_loader,
    distort_schedulers,
    frag_occurance_source="train",
    optimizer=None,
    is_not_noised=False,
):
    (
        node_distort_scheduler,
        edge_distort_scheduler,
        cont_distort_scheduler,
    ) = distort_schedulers
    st = time.time()
    total_loss, total_mol_loss, total_frag_loss = 0.0, 0.0, 0.0
    total_mol_mae, total_frag_mae = 0.0, 0.0
    all_frag_props = []
    n_batch = 1

    for graph, coarse_graph in tqdm(data_loader):
        graph.to("cuda")
        coarse_graph.to("cuda")
        bs = coarse_graph.batch.max() + 1
        device = graph.h.device
        n_node = coarse_graph.h.size(0)
        n_edge = coarse_graph.full_e.size(0)

        # sample fragments from predefined distribution
        base_frag_idxs = data_loader.dataset.random_select_frags_by_occurance(
            n=fm_cfg.n_base_frag,
            split=frag_occurance_source,
            temperature=fm_cfg.frag_select_temp,
        ).to(device)
        exst_frag_idxs = torch.unique(coarse_graph.h)
        cur_frag_idxs = torch.unique(torch.cat([base_frag_idxs, exst_frag_idxs], dim=0))
        is_base_frag_mask = torch.isin(cur_frag_idxs, base_frag_idxs)
        n_cur_frag = len(cur_frag_idxs)
        del base_frag_idxs
        del exst_frag_idxs

        # make fragment mask to ensure independant over data in batch
        frag_mask = torch.zeros(bs, n_cur_frag).long().to(device)
        frag_mask[:, is_base_frag_mask] = 1  # include all base frags (negs)
        frag_mask = frag_mask[coarse_graph.batch]  # [n_node, n_cur_frag]
        temp_h_type = torch.searchsorted(cur_frag_idxs, coarse_graph.h)  # [n_node]
        temp_h_onehot = F.one_hot(temp_h_type, num_classes=n_cur_frag).float()
        temp_h_in_batch = scatter(
            temp_h_onehot, coarse_graph.batch, reduce="sum", dim=0
        )
        temp_h_in_batch = temp_h_in_batch[coarse_graph.batch].bool()  # [n_node]
        frag_mask = frag_mask.bool() | temp_h_in_batch  # [bs, n_cur_frag]
        del temp_h_type, temp_h_onehot, temp_h_in_batch

        # get fragment graphs
        cur_frag_graphs = data_loader.dataset.get_frags_from_index_tensor(cur_frag_idxs)
        cur_frag_graphs.to(device)

        # encode latent variable
        with torch.no_grad():
            z, _ = ae_model.encode(
                graph.h,
                graph.h_junction_count,
                graph.h_in_frag_label,
                graph.h_aux_frag_label,
                graph.e_index,
                graph.e,
                graph.batch,
            )
        z = z.detach()

        # transform latent variables
        if fm_cfg.latent_transform == "min_max":
            min_z = cfg.latent_transform_param["min"].unsqueeze(0).cuda()
            max_z = cfg.latent_transform_param["max"].unsqueeze(0).cuda()
            z = 2 * (z - min_z) / (max_z - min_z) - 1.0
        elif cfg.latent_transform == "leave":
            pass
        else:
            raise NotImplementedError

        # randomly sample training time and convert to noising time
        model_t = torch.rand(bs).to(device)
        model_t[bs // 2 :] = 1.0 - model_t[: bs // 2]
        if is_not_noised:
            model_t = torch.ones_like(model_t)
        node_t = node_distort_scheduler.convert_time(model_t)
        edge_t = edge_distort_scheduler.convert_time(model_t)
        cont_t = cont_distort_scheduler.convert_time(model_t)

        # convert global node index to inbatch node index
        h_type = torch.searchsorted(cur_frag_idxs, coarse_graph.h)
        e_type = coarse_graph.full_e

        # noise node type
        if cfg.node_prior == "mask":
            h_prior_type = torch.ones(n_node).to(device).long() * n_cur_frag
            ht_type = h_type.clone()
            crpt_h_mask_prob = (1 - node_t)[coarse_graph.batch]
            crpt_h_mask = torch.rand_like(crpt_h_mask_prob) < crpt_h_mask_prob
            ht_type[crpt_h_mask] = h_prior_type[crpt_h_mask]
        elif cfg.node_prior == "uniform":
            h_prior_prob = torch.ones(n_node, n_cur_frag).to(device)
            h_prior_prob = h_prior_prob * frag_mask / frag_mask.sum(dim=1, keepdim=True)
            h_prior_type = sample_from_prob(h_prior_prob)
            ht_type = h_type.clone()
            crpt_h_mask_prob = (1 - node_t)[coarse_graph.batch]
            crpt_h_mask = torch.rand_like(crpt_h_mask_prob) < crpt_h_mask_prob
            ht_type[crpt_h_mask] = h_prior_type[crpt_h_mask]
        else:
            raise NotImplementedError

        # noise edge type
        if cfg.edge_prior == "mask":  # needs additional type
            e_prior_type = torch.ones(n_edge).to(device).long() * 2
            et_type = e_type.clone()
            crpt_e_mask_prob = (1 - edge_t)[
                coarse_graph.batch[coarse_graph.full_e_index[0]]
            ]
            crpt_e_mask = torch.rand_like(crpt_e_mask_prob) < crpt_e_mask_prob
            et_type[crpt_e_mask] = e_prior_type[crpt_e_mask]
        elif cfg.edge_prior == "absorb":
            e_prior_type = torch.zeros(n_edge).to(device).long()
            et_type = e_type.clone()
            crpt_e_mask_prob = (1 - edge_t)[
                coarse_graph.batch[coarse_graph.full_e_index[0]]
            ]
            crpt_e_mask = torch.rand_like(crpt_e_mask_prob) < crpt_e_mask_prob
            et_type[crpt_e_mask] = e_prior_type[crpt_e_mask]
        elif cfg.edge_prior == "uniform":
            e_prior_prob = torch.ones(n_edge, 2).to(device) / 2
            e_prior_type = sample_from_prob(e_prior_prob)
            et_type = e_type.clone()
            crpt_e_mask_prob = (1 - edge_t)[
                coarse_graph.batch[coarse_graph.full_e_index[0]]
            ]
            crpt_e_mask = torch.rand_like(crpt_e_mask_prob) < crpt_e_mask_prob
            et_type[crpt_e_mask] = e_prior_type[crpt_e_mask]
        else:
            raise NotImplementedError

        # noise latent z
        z_prior = torch.randn_like(z).to(device)
        zt = z_prior + (z - z_prior) * cont_t.unsqueeze(1)

        # change to onehot
        if cfg.node_prior == "mask":
            ht_onehot = F.one_hot(ht_type, num_classes=n_cur_frag + 1).float()
        else:
            ht_onehot = F.one_hot(ht_type, num_classes=n_cur_frag).float()
        if cfg.edge_prior == "mask":
            et_onehot = F.one_hot(et_type, num_classes=3).float()
        else:
            et_onehot = F.one_hot(et_type, num_classes=2).float()

        # process the fragments to latent variables
        if cfg.node_prior == "mask":
            get_mask_node_embd = True
        else:
            get_mask_node_embd = False
        if cfg.frag_fully_connected_graph:
            frag_zs, pred_frag_prop = frag_embedder(
                cur_frag_graphs.h,
                cur_frag_graphs.h_junction_count,
                cur_frag_graphs.e_index,
                cur_frag_graphs.e,
                cur_frag_graphs.g,
                cur_frag_graphs.batch,
                cat_mask=get_mask_node_embd,
            )
        else:
            frag_zs, pred_frag_prop = frag_embedder(
                cur_frag_graphs.h,
                cur_frag_graphs.h_junction_count,
                cur_frag_graphs.full_e_index,
                cur_frag_graphs.full_e,
                cur_frag_graphs.g,
                cur_frag_graphs.batch,
                cat_mask=get_mask_node_embd,
            )
        pred_frag_mu = pred_frag_prop[:, 0]

        # make prediction logits
        pred_mol_prop = coarse_gnn(
            ht_onehot,
            coarse_graph.full_e_index,
            et_onehot,
            zt,
            coarse_graph.batch,
            model_t,
            frag_zs,
        )
        pred_mol_mu = pred_mol_prop[:, 0]

        # fragment level loss
        pred_frag_mu_ = pred_frag_mu[h_type]
        ans_frag_prop_ = coarse_graph.prop[coarse_graph.batch]
        if "smina" in cfg.calc_prop:
            is_data_mask = ans_frag_prop_ < 0
        else:
            is_data_mask = torch.ones_like(ans_frag_prop_).bool()
        frag_prop_loss = F.mse_loss(
            pred_frag_mu_[is_data_mask],
            ans_frag_prop_[is_data_mask],
            reduction="mean",
        )
        all_frag_props += ans_frag_prop_.cpu().tolist()
        frag_prop_mae = F.l1_loss(pred_frag_mu_, ans_frag_prop_, reduction="mean")

        # molecule level loss
        if "smina" in cfg.calc_prop:
            is_data_mask = coarse_graph.prop < 0
        else:
            is_data_mask = torch.ones_like(coarse_graph.prop).bool()
        mol_prop_loss = F.mse_loss(
            pred_mol_mu[is_data_mask],
            coarse_graph.prop[is_data_mask],
            reduction="mean",
        )
        mol_prop_mae = F.l1_loss(pred_mol_mu, coarse_graph.prop, reduction="mean")

        # merge loss
        loss = mol_prop_loss * cfg.mol_prop_loss + frag_prop_loss * cfg.frag_prop_loss
        total_loss += loss.item()
        total_mol_loss += mol_prop_loss.item()
        total_mol_mae += mol_prop_mae.cpu().item()
        total_frag_loss += frag_prop_loss.item()
        total_frag_mae += frag_prop_mae.cpu().item()
        n_batch += 1

        # optimize model
        if optimizer != None:
            # lr adjustment
            if (
                (not cfg.is_resume)
                and cfg.lr_warmup
                and (cfg.n_iter_done < cfg.lr_warmup_iter)
            ):
                current_lr = cfg.lr * ((cfg.n_iter_done + 1) / cfg.lr_warmup_iter)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                # print(param_group["lr"], end=" ", flush=True)
            else:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cfg.lr

            # optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(frag_embedder.parameters(), cfg.grad_clip)
            optimizer.step()
            cfg.n_iter_done += 1

            # mea
            if cfg.use_ema:
                update_ema(frag_embedder, ema_frag_embedder)
                update_ema(coarse_gnn, ema_coarse_gnn)

    # out for batch
    result = {}
    result["loss"] = total_loss / n_batch
    result["mol_loss"] = total_mol_loss / n_batch
    result["mol_mae"] = total_mol_mae / n_batch
    result["frag_loss"] = total_frag_loss / n_batch
    result["frag_mae"] = total_frag_mae / n_batch
    result["time"] = time.time() - st
    # get the average MAE against the true frag prop
    all_frag_props = np.array(all_frag_props)
    result["expected_frag_mae"] = np.mean(
        np.abs(all_frag_props - np.mean(all_frag_props))
    )
    return result


if __name__ == "__main__":
    cfg_fn = sys.argv[1]
    cfg = read_yaml_as_easydict(cfg_fn)
    # cfg.use_frag_valency = False
    # check if cfg has "is_clean_only
    if not "is_clean_only" in cfg:
        cfg.is_clean_only = False

    # if resuming...
    if "start_dirn" in cfg:
        prev_cfg = read_yaml_as_easydict(os.path.join(cfg.start_dirn, "cfg.yaml"))
        prev_cfg.update(cfg)  # override newly given cfgs
        cfg = prev_cfg
        cfg.is_resume = True
    else:
        cfg.is_resume = False

    pprint(cfg)

    # get flow model cfg
    fm_cfg_fn = os.path.join(cfg.fm_model_dirn, "cfg.yaml")
    fm_cfg = read_yaml_as_easydict(fm_cfg_fn)

    # get trained ae model and fm cfg
    ae_cfg_fn = os.path.join(fm_cfg.ae_model_dirn, "cfg.yaml")
    ae_cfg = read_yaml_as_easydict(ae_cfg_fn)
    cfg.latent_z_dim = ae_cfg.latent_z_dim

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if not cfg.is_resume:
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        cfg.save_dirn = f"save/disc_model/{cur_time}_{cfg.tag}/"
    else:
        bn = os.path.basename(cfg.start_dirn)  #
        cfg.save_dirn = f"save/disc_model/{bn}_re/"
        print(f"Training is resumed and saved to {cfg.save_dirn}")

    # wandb
    if cfg.use_wandb:
        if not cfg.is_resume:
            name = f"{cur_time}_{cfg.tag}"
        else:
            name = f"{bn}_re"
        wandb.init(project="FragFM_disc", name=name, config=cfg)

    os.makedirs(cfg.save_dirn, exist_ok=True)
    log_cfg_fn = os.path.join(cfg.save_dirn, "cfg.yaml")
    write_easydict_as_yaml(cfg, log_cfg_fn)

    # get distortion scheduler
    node_distort_scheduler = DistortScheduler(fm_cfg.node_distort_schedule)
    edge_distort_scheduler = DistortScheduler(fm_cfg.edge_distort_schedule)
    cont_distort_scheduler = DistortScheduler(fm_cfg.latent_z_distort_schedule)
    distort_schedulers = [
        node_distort_scheduler,
        edge_distort_scheduler,
        cont_distort_scheduler,
    ]

    train_set = FragFMDataset(
        lmdb_fn=cfg.data_dirn,
        frag_lmdb_fn=cfg.frag_data_dirn,
        frag_smi_to_idx_fn=cfg.frag_smi_to_idx_fn,
        data_split="train",
        debug=cfg.debug,
        calc_prop=cfg.calc_prop,
    )
    valid_set = FragFMDataset(
        lmdb_fn=cfg.data_dirn,
        frag_lmdb_fn=cfg.frag_data_dirn,
        frag_smi_to_idx_fn=cfg.frag_smi_to_idx_fn,
        data_split="valid",
        debug=cfg.debug,
        calc_prop=cfg.calc_prop,
    )
    test_set = FragFMDataset(
        lmdb_fn=cfg.data_dirn,
        frag_lmdb_fn=cfg.frag_data_dirn,
        frag_smi_to_idx_fn=cfg.frag_smi_to_idx_fn,
        data_split="test",
        debug=cfg.debug,
        calc_prop=cfg.calc_prop,
    )

    # smapler loader
    train_sampler = RandomSampler(train_set, num_samples=cfg.bs * cfg.log_every_n_iter)
    valid_sampler = RandomSampler(valid_set, num_samples=cfg.bs * cfg.log_every_n_iter)
    test_sampler = RandomSampler(test_set, num_samples=cfg.bs * cfg.log_every_n_iter)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.bs,
        sampler=train_sampler,
        collate_fn=collate_frag_fm_dataset,
        num_workers=cfg.n_worker,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.bs,
        sampler=valid_sampler,
        collate_fn=collate_frag_fm_dataset,
        num_workers=cfg.n_worker,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.bs,
        sampler=test_sampler,
        collate_fn=collate_frag_fm_dataset,
        num_workers=cfg.n_worker,
    )

    # set ae model
    ae_model = FragJunctionAE(ae_cfg)
    ae_model.load_state_dict(
        torch.load(
            os.path.join(fm_cfg.ae_model_dirn, "model_best.pt"), map_location="cpu"
        )
    )
    ae_model.eval()
    ae_model.cuda()

    # get latent transform
    with open(os.path.join(cfg.fm_model_dirn, "latent_transform_param.pkl"), "rb") as f:
        cfg.latent_transform_param = pickle.load(f)
    print("Latent transition paramter loaded:")
    print(cfg.latent_transform_param)

    # frag embedder
    cfg.node_prior, cfg.edge_prior = fm_cfg.node_prior, fm_cfg.edge_prior
    frag_embedder = FragToVectReadout(cfg)
    frag_embedder.cuda()
    print(f"Trainable parameters: {sum(p.numel() for p in frag_embedder.parameters())}")

    # coarse gnn
    cfg.latent_z_dim = ae_cfg.latent_z_dim
    coarse_gnn = CoarseGraphReadout(cfg)
    coarse_gnn.cuda()
    print(f"Trainable parameters: {sum(p.numel() for p in coarse_gnn.parameters())}")

    # load if resume
    if cfg.is_resume:
        frag_embedder.load_state_dict(
            torch.load(os.path.join(cfg.start_dirn, f"frag_embedder_best.pt"))
        )
        coarse_gnn.load_state_dict(
            torch.load(os.path.join(cfg.start_dirn, f"coarse_propagate_best.pt"))
        )
        print("Resuming, model loaded")
    else:
        print("Training from scratch")

    # ema
    if cfg.use_ema:
        ema_frag_embedder = copy.deepcopy(frag_embedder)
        ema_coarse_gnn = copy.deepcopy(coarse_gnn)
        for param in ema_frag_embedder.parameters():
            param.requires_grad = False
        for param in ema_coarse_gnn.parameters():
            param.requires_grad = False
        ema_frag_embedder.eval()
        ema_coarse_gnn.eval()
        print("EMA model created")

    # optimizer
    params = list(frag_embedder.parameters()) + list(coarse_gnn.parameters())
    # optimzer
    if cfg.optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.optimizer_betas,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.optimizer_betas,
            weight_decay=cfg.weight_decay,
            amsgrad=True,
        )
    pprint(optimizer)

    # load if resume
    if cfg.is_resume:
        optimizer.load_state_dict(
            torch.load(os.path.join(cfg.start_dirn, f"optimizer_best.pt"))
        )
        print("Resuming, optimizer loaded")
    else:
        pass

    best_valid_loss, cor_test_loss, cfg.n_iter_done = 9.0e9, 9.0e9, 0
    for epoch_idx in range(1, 1000000000):
        print(epoch_idx)

        # train
        frag_embedder.train(), coarse_gnn.train()
        train_result = process_single_epoch(
            cfg,
            ae_model,
            frag_embedder,
            coarse_gnn,
            train_loader,
            distort_schedulers,
            frag_occurance_source="train",
            optimizer=optimizer,
            is_not_noised=cfg.is_clean_only,
        )
        train_result = add_prefix_to_dict_key(train_result, "train_")
        train_result["epoch"] = epoch_idx
        if cfg.use_wandb:
            wandb.log(train_result)
        pprint(train_result)

        if epoch_idx % cfg.eval_every_n_log == 0:
            # valid
            frag_embedder.eval(), coarse_gnn.eval()
            with torch.no_grad():
                if cfg.use_ema:
                    valid_result = process_single_epoch(
                        cfg,
                        ae_model,
                        ema_frag_embedder,
                        ema_coarse_gnn,
                        valid_loader,
                        distort_schedulers,
                        frag_occurance_source="valid",
                        is_not_noised=cfg.is_clean_only,
                    )
                else:
                    valid_result = process_single_epoch(
                        cfg,
                        ae_model,
                        frag_embedder,
                        coarse_gnn,
                        valid_loader,
                        distort_schedulers,
                        frag_occurance_source="valid",
                        is_not_noised=cfg.is_clean_only,
                    )
            valid_result = add_prefix_to_dict_key(valid_result, prefix="valid_")
            valid_result["epoch"] = epoch_idx

            if valid_result["valid_loss"] < best_valid_loss:
                best_tag = True
                best_valid_loss = valid_result["valid_loss"]
                torch.save(
                    frag_embedder.state_dict(),
                    os.path.join(cfg.save_dirn, f"frag_embedder_best.pt"),
                )
                torch.save(
                    coarse_gnn.state_dict(),
                    os.path.join(cfg.save_dirn, f"coarse_propagate_best.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(cfg.save_dirn, f"optimizer_best.pt"),
                )
                # save ema
                if cfg.use_ema:
                    torch.save(
                        ema_frag_embedder.state_dict(),
                        os.path.join(cfg.save_dirn, f"frag_embedder_ema_best.pt"),
                    )
                    torch.save(
                        ema_coarse_gnn.state_dict(),
                        os.path.join(cfg.save_dirn, f"coarse_propagate_ema_best.pt"),
                    )
            else:
                best_tag = False

            valid_result["best_valid_loss"] = best_valid_loss
            if cfg.use_wandb:
                wandb.log(valid_result)
            pprint(valid_result)

            # test
            frag_embedder.eval(), coarse_gnn.eval()
            with torch.no_grad():
                if cfg.use_ema:
                    test_result = process_single_epoch(
                        cfg,
                        ae_model,
                        ema_frag_embedder,
                        ema_coarse_gnn,
                        test_loader,
                        distort_schedulers,
                        frag_occurance_source="test",
                        is_not_noised=cfg.is_clean_only,
                    )
                else:
                    test_result = process_single_epoch(
                        cfg,
                        ae_model,
                        frag_embedder,
                        coarse_gnn,
                        test_loader,
                        distort_schedulers,
                        frag_occurance_source="test",
                        is_not_noised=cfg.is_clean_only,
                    )
            test_result = add_prefix_to_dict_key(test_result, prefix="test_")
            test_result["epoch"] = epoch_idx
            if best_tag:
                cor_test_loss = test_result["test_loss"]
            test_result["corresponding_test_loss"] = cor_test_loss
            if cfg.use_wandb:
                wandb.log(test_result)
            pprint(test_result)

            # check test for time=1
            frag_embedder.eval(), coarse_gnn.eval()
            with torch.no_grad():
                if cfg.use_ema:
                    test_clean_result = process_single_epoch(
                        cfg,
                        ae_model,
                        ema_frag_embedder,
                        ema_coarse_gnn,
                        test_loader,
                        distort_schedulers,
                        frag_occurance_source="test",
                        is_not_noised=True,
                    )
                else:
                    test_clean_result = process_single_epoch(
                        cfg,
                        ae_model,
                        frag_embedder,
                        coarse_gnn,
                        test_loader,
                        distort_schedulers,
                        frag_occurance_source="test",
                        is_not_noised=True,
                    )
            test_clean_result = add_prefix_to_dict_key(
                test_clean_result, prefix="test_clean_"
            )
            test_clean_result["epoch"] = epoch_idx
            if cfg.use_wandb:
                wandb.log(test_clean_result)
            pprint(test_clean_result)

            # save every iter
            torch.save(
                frag_embedder.state_dict(),
                os.path.join(
                    cfg.save_dirn,
                    f"frag_embedder_{epoch_idx * cfg.log_every_n_iter}.pt",
                ),
            )
            torch.save(
                coarse_gnn.state_dict(),
                os.path.join(
                    cfg.save_dirn,
                    f"coarse_propagate_{epoch_idx * cfg.log_every_n_iter}.pt",
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    cfg.save_dirn,
                    f"optimizer_{epoch_idx * cfg.log_every_n_iter}.pt",
                ),
            )
            # save EMA
            if cfg.use_ema:
                torch.save(
                    ema_frag_embedder.state_dict(),
                    os.path.join(
                        cfg.save_dirn,
                        f"frag_embedder_ema_{epoch_idx * cfg.log_every_n_iter}.pt",
                    ),
                )
                torch.save(
                    ema_coarse_gnn.state_dict(),
                    os.path.join(
                        cfg.save_dirn,
                        f"coarse_propagate_ema_{epoch_idx * cfg.log_every_n_iter}.pt",
                    ),
                )

        print()

    if cfg.use_wandb:
        wandb.finish()
