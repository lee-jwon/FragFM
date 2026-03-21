import os
import random
import shutil
import sys
import time
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Draw
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch_geometric.utils import scatter
from tqdm import tqdm

from fragfm.dataset import FragJunctionAEDataset, collate_frag_junction_ae_dataset
from fragfm.model.ae import FragJunctionAE
from fragfm.utils.file import *
from fragfm.utils.file import add_prefix_to_dict_key, read_yaml_as_easydict
from fragfm.utils.graph_ops import *


def process_single_epoch(epoch_idx, cfg, model, data_loader, optimizer=None):
    st = time.time()
    total_loss, total_recon_loss, total_reg_loss = 0.0, 0.0, 0.0
    n_edge, n_graph = 0, 0
    n_correct_edge, n_correct_graph = 0, 0
    n_batch = 0
    result = {}
    for graph in tqdm(data_loader):
        graph.to("cuda")
        model_out = model(
            graph.h,
            graph.h_junction_count,
            graph.h_in_frag_label,
            graph.h_aux_frag_label,
            graph.e_index,
            graph.e,
            graph.decomp_e_index,
            graph.decomp_e,
            graph.ae_to_pred_index,
            graph.batch,
        )
        pred_jxn = torch.sigmoid(model_out["e"])
        recon_loss = F.binary_cross_entropy(
            pred_jxn, graph.ae_to_pred.float(), reduction="mean"
        )  # per

        reg_loss = -0.5 * torch.mean(
            1
            + model_out["z_logvar"]
            - model_out["z_mu"].pow(2)
            - model_out["z_logvar"].exp()
        )  # per digit

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

        # Accumulate loss and accuracy
        loss = recon_loss + reg_loss * cfg.reg_loss
        total_recon_loss += recon_loss.item()
        total_reg_loss += reg_loss.item()
        total_loss += loss.item()
        n_batch += 1

        # update params
        if optimizer != None:
            if (
                (not cfg.is_resume)
                and cfg.lr_warmup
                and (cfg.n_iter_done < cfg.lr_warmup_iter)
            ):
                current_lr = cfg.lr * ((cfg.n_iter_done + 1) / cfg.lr_warmup_iter)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                # print(param_group["lr"], end=" ", flush=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            cfg.n_iter_done += 1

    result["recon_loss"] = total_recon_loss / n_batch
    result["reg_loss"] = total_reg_loss / n_batch
    result["total_loss"] = total_loss / n_batch
    result["edge_accuracy"] = n_correct_edge / n_edge
    result["graph_accuracy"] = n_correct_graph / n_graph
    result["time"] = time.time() - st
    return result


if __name__ == "__main__":
    cfg_fn = sys.argv[1]
    cfg = read_yaml_as_easydict(cfg_fn)

    # if resuming...
    if "start_dirn" in cfg:
        prev_cfg = read_yaml_as_easydict(os.path.join(cfg.start_dirn, "cfg.yaml"))
        prev_cfg.update(cfg)  # override newly given cfgs
        cfg = prev_cfg
        cfg.is_resume = True
    else:
        cfg.is_resume = False

    pprint(cfg)

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
        cfg.save_dirn = f"save/ae_model/{cur_time}_{cfg.tag}/"
    else:
        bn = os.path.basename(cfg.start_dirn)  #
        cfg.save_dirn = f"save/ae_model/{bn}_re/"
        print(f"Training is resumed and saved to {cfg.save_dirn}")

    # cur_time = datetime.now().strftime("%y%m%d_%H%M")
    # cfg.save_dirn = f"save/ae_model/{cur_time}_{cfg.tag}/"

    # wandb
    if cfg.use_wandb:
        import wandb

        if not cfg.is_resume:
            name = f"{cur_time}_{cfg.tag}"
        else:
            name = f"{bn}_re"
        wandb.init(project="FragFM_ae", name=name, config=cfg)

    os.makedirs(cfg.save_dirn, exist_ok=True)
    # log_frag_embedder_ckpt_fn = os.path.join(cfg.save_dirn, "frag_embd.pt")
    # log_coarse_propagate_ckpt_fn = os.path.join(cfg.save_dirn, "coarse_propagate.pt")
    log_cfg_fn = os.path.join(cfg.save_dirn, "cfg.yaml")
    # shutil.copy(cfg_fn, log_cfg_fn)
    write_easydict_as_yaml(cfg, log_cfg_fn)

    # load dataset
    train_set = FragJunctionAEDataset(
        cfg.data_dirn, data_split="train", debug=cfg.debug
    )
    valid_set = FragJunctionAEDataset(
        cfg.data_dirn, data_split="valid", debug=cfg.debug
    )
    test_set = FragJunctionAEDataset(cfg.data_dirn, data_split="test", debug=cfg.debug)
    print("Load dataset")
    print(f"{len(train_set)} / {len(valid_set)} / {len(test_set)}")

    # sampler loader
    train_sampler = RandomSampler(train_set, num_samples=cfg.bs * cfg.log_every_n_iter)
    valid_sampler = RandomSampler(valid_set, num_samples=cfg.bs * cfg.log_every_n_iter)
    test_sampler = RandomSampler(test_set, num_samples=cfg.bs * cfg.log_every_n_iter)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.bs,
        sampler=train_sampler,
        collate_fn=collate_frag_junction_ae_dataset,
        num_workers=cfg.n_worker,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.bs,
        sampler=valid_sampler,
        collate_fn=collate_frag_junction_ae_dataset,
        num_workers=cfg.n_worker,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.bs,
        sampler=test_sampler,
        collate_fn=collate_frag_junction_ae_dataset,
        num_workers=cfg.n_worker,
    )

    # model
    model = FragJunctionAE(cfg)
    model.cuda()
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # load if resume
    if cfg.is_resume:
        model.load_state_dict(
            torch.load(os.path.join(cfg.start_dirn, f"model_best.pt"))
        )
        print("Resuming, model loaded")
    else:
        print("Training from scratch")

    # optimzer
    if cfg.optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.optimizer_betas,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.optimizer_betas,
            weight_decay=cfg.weight_decay,
        )
    pprint(optimizer)

    if cfg.is_resume:
        optimizer.load_state_dict(
            torch.load(os.path.join(cfg.start_dirn, f"optimizer_best.pt"))
        )
        print("Resuming, optimizer loaded")
    else:
        pass

    best_valid_acc, cor_test_acc, cfg.n_iter_done = 0.0, 0.0, 0
    for epoch_idx in range(1, 1000000000):
        print(epoch_idx)

        # train
        model.train()
        train_result = process_single_epoch(
            epoch_idx, cfg, model, train_loader, optimizer
        )
        train_result = add_prefix_to_dict_key(train_result, prefix="train_")
        train_result["epoch"] = epoch_idx
        if cfg.use_wandb:
            wandb.log(train_result)
        pprint(train_result)

        if epoch_idx % cfg.eval_every_n_log == 0:
            # valid
            model.eval()
            with torch.no_grad():
                valid_result = process_single_epoch(epoch_idx, cfg, model, valid_loader)
            valid_result = add_prefix_to_dict_key(valid_result, prefix="valid_")
            valid_result["epoch"] = epoch_idx
            if valid_result["valid_graph_accuracy"] > best_valid_acc:
                best_tag = True
                best_valid_acc = valid_result["valid_graph_accuracy"]
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.save_dirn, f"model_best.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(cfg.save_dirn, f"optimizer_best.pt"),
                )
            else:
                best_tag = False
            valid_result["best_valid_accuracy"] = best_valid_acc
            if cfg.use_wandb:
                wandb.log(valid_result)
            pprint(valid_result)

            # test
            model.eval()
            with torch.no_grad():
                test_result = process_single_epoch(epoch_idx, cfg, model, test_loader)
            test_result = add_prefix_to_dict_key(test_result, prefix="test_")
            test_result["epoch"] = epoch_idx
            if best_tag:
                cor_test_acc = test_result["test_graph_accuracy"]
            test_result["corresponding_test_accuracy"] = cor_test_acc
            if cfg.use_wandb:
                wandb.log(test_result)
            pprint(test_result)

            # save every iter
            torch.save(
                model.state_dict(),
                os.path.join(
                    cfg.save_dirn, f"model_{epoch_idx * cfg.log_every_n_iter}.pt"
                ),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    cfg.save_dirn,
                    f"optimizer_{epoch_idx * cfg.log_every_n_iter}.pt",
                ),
            )

        print()

    if cfg.use_wandb:
        wandb.finish()
