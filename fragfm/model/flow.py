import torch
from easydict import EasyDict
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from fragfm.model.gt_digress import XEyTransformerLayer
from fragfm.model.layer import *
from fragfm.model.mpnn import GlobalMPNNLayer2


class FragToVect(nn.Module):
    def __init__(self, cfg: EasyDict):
        super().__init__()
        self.cfg = cfg

        # trianable mask type embedding
        self.mask_type_frag_z = nn.Parameter(
            torch.zeros(1, cfg.embd_h_dim), requires_grad=False
        )

        self.embd_h = nn.Embedding(300, cfg.embd_h_dim // 2)
        self.embd_h_junction_count = nn.Embedding(
            5, cfg.embd_h_dim // 2 + cfg.embd_h_dim % 2
        )
        self.embd_e = nn.Embedding(5, cfg.embd_e_dim)
        self.embd_g = MLP(
            [cfg.frag_g_dim, cfg.embd_h_dim], init_method=cfg.fc_init_method
        )

        if cfg.in_frag_rrwp_walk_length > 0:
            self.merge_embd_e = MLP(
                [cfg.embd_e_dim + cfg.in_frag_rrwp_walk_length, cfg.embd_e_dim],
                init_method=cfg.fc_init_method,
            )
        else:
            pass

        # layers
        layers = []
        for _ in range(cfg.backbone_n_frag_to_vect_layer):
            layers.append(
                GlobalMPNNLayer2(
                    h_dim=cfg.embd_h_dim,
                    e_dim=cfg.embd_e_dim,
                    g_dim=cfg.embd_h_dim,
                    hid_dim=cfg.hid_dim,
                    dropout=cfg.backbone_dropout,
                    layer_norm=cfg.backbone_layer_norm,
                    activation=cfg.backbone_activation,
                    init_method=cfg.backbone_init_method,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.readout_frag_latent = MLP(
            [cfg.embd_h_dim, cfg.embd_h_dim, cfg.embd_h_dim],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

    def forward(self, h, h_junction_count, e_index, e, g, batch, cat_mask=False):
        h1 = self.embd_h(h)
        h2 = self.embd_h(h_junction_count)
        h_embd = torch.cat([h1, h2], dim=1)
        e_embd = self.embd_e(e)
        bidi_e_index, bidi_e_embd = half_edge_to_full_edge(e_index, e_embd)

        if self.cfg.in_frag_rrwp_walk_length > 0:
            _, bidi_e = half_edge_to_full_edge(e_index, e)
            _, e_rrwp = compute_degree_and_rrwp(
                bidi_e_index,
                bidi_e,
                h_embd.size(0),
                walk_length=self.cfg.in_frag_rrwp_walk_length,
            )
            bidi_e_embd = torch.cat([bidi_e_embd, e_rrwp], dim=1)
            bidi_e_embd = self.merge_embd_e(bidi_e_embd)

        g_embd = self.embd_g(g)

        for layer in self.layers:
            h_embd, bidi_e_embd, g_embd = layer(
                h_embd, bidi_e_index, bidi_e_embd, g_embd, batch
            )

        z = self.readout_frag_latent(g_embd)  # [n_frag, hid_dim], for each inbatch type
        if cat_mask:
            z = torch.cat([z, self.mask_type_frag_z], dim=0)  # add it to last!
        return z


class CoarseGraphPropagate(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embd_latent_z = MLP(
            [cfg.latent_z_dim, cfg.embd_h_dim], init_method=cfg.fc_init_method
        )
        self.embd_timestep = MLP([1, cfg.embd_h_dim], init_method=cfg.fc_init_method)

        if self.cfg.edge_prior == "mask":
            self.embd_coarse_e = MLP(
                [3, cfg.embd_e_dim], init_method=cfg.fc_init_method
            )
        else:
            self.embd_coarse_e = MLP(
                [2, cfg.embd_e_dim], init_method=cfg.fc_init_method
            )

        # valency embedder
        if "use_frag_valency" not in cfg:
            self.cfg.use_frag_valency = False
        if cfg.use_frag_valency:
            self.embd_coarse_h_valency = nn.Embedding(21, cfg.embd_h_dim)
            self.merge_embd_coarse_h = MLP(
                [cfg.embd_h_dim * 2, cfg.embd_h_dim], init_method=cfg.fc_init_method
            )
        else:
            pass

        # RRWP embedder
        if cfg.rrwp_walk_length > 0:
            self.merge_embd_e = MLP(
                [cfg.embd_e_dim + cfg.rrwp_walk_length, cfg.embd_e_dim],
                init_method=cfg.fc_init_method,
            )
        else:
            pass

        # global feature embedder
        self.merge_embd_g = MLP(
            [cfg.embd_h_dim * 3, cfg.embd_h_dim, cfg.embd_h_dim],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

        # fragment bag embedder (optional)
        if cfg.embd_frag_bag_type == "attention":
            self.embd_frag_g_att = MLP(
                [cfg.embd_h_dim, cfg.embd_h_dim, 1],
                dropout=cfg.fc_dropout,
                layer_norm=cfg.fc_layer_norm,
                activation=cfg.fc_activation,
                init_method=cfg.fc_init_method,
            )

        # layers
        layers = []
        for _ in range(cfg.backbone_n_coarse_graph_propagate_layer):
            if cfg.backbone_type == "mpnn":
                layers.append(
                    GlobalMPNNLayer2(
                        h_dim=cfg.embd_h_dim,
                        e_dim=cfg.embd_e_dim,
                        g_dim=cfg.embd_h_dim,
                        hid_dim=cfg.hid_dim,
                        dropout=cfg.backbone_dropout,
                        layer_norm=cfg.backbone_layer_norm,
                        activation=cfg.backbone_activation,
                        init_method=cfg.backbone_init_method,
                    )
                )
            elif cfg.backbone_type == "gt_digress":
                layers.append(
                    XEyTransformerLayer(
                        dx=cfg.embd_h_dim,
                        de=cfg.embd_e_dim,
                        dy=cfg.embd_h_dim,
                        n_head=cfg.backbone_n_head,
                    )
                )
            else:
                raise NotImplementedError
        self.layers = nn.ModuleList(layers)

        self.fc_readout_h = MLP(
            [cfg.embd_h_dim, cfg.embd_h_dim, cfg.embd_h_dim],
            dropout=cfg.fc_dropout,
            layer_norm=False,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

        # edge readout
        self.fc_readout_e = MLP(
            [cfg.embd_e_dim, cfg.embd_e_dim, 2],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

        # latent z readout
        self.fc_readout_latent_z = MLP(
            [cfg.embd_h_dim, cfg.embd_h_dim, cfg.latent_z_dim],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

    def forward(
        self,
        coarse_h_prob,
        e_index,
        e,
        z,
        batch,
        timestep,
        frag_zs,
        coarse_h_valency=None,
    ):
        n_node = coarse_h_prob.size(0)
        n_edge = e.size(0)
        n_frag = frag_zs.size(0)
        bs = batch.max() + 1
        device = frag_zs.device

        # make fragment bag embedding
        if self.cfg.embd_frag_bag_type == "sum":
            frag_bag_embd = frag_zs.sum(dim=0) / 100
        elif self.cfg.embd_frag_bag_type == "mean":
            frag_bag_embd = frag_zs.mean(dim=0)
        elif self.cfg.embd_frag_bag_type == "attention":
            frag_zs_score = self.embd_frag_g_att(frag_zs).squeeze(1)
            frag_zs_att = torch.softmax(frag_zs_score, dim=0)
            frag_bag_embd = (frag_zs_att.unsqueeze(1) * frag_zs).sum(dim=0)
        elif self.cfg.embd_frag_bag_type == "mask":
            frag_bag_embd = (
                torch.zeros(
                    frag_zs.size(1),
                )
                .to(device)
                .detach()
            )
        else:
            raise NotImplementedError

        # set node type
        h_embd = coarse_h_prob.matmul(frag_zs)

        # when given valency feature, merge
        if self.cfg.use_frag_valency:
            coarse_h_valency = torch.clamp(coarse_h_valency, min=-10, max=10) + 10
            coarse_h_valency_embd = self.embd_coarse_h_valency(coarse_h_valency)
            h_embd = torch.cat([h_embd, coarse_h_valency_embd], dim=1)
            h_embd = self.merge_embd_coarse_h(h_embd)
        else:
            pass

        # make edge bidirectral
        e_embd = self.embd_coarse_e(e)
        bidi_e_index, bidi_e_embd = half_edge_to_full_edge(e_index, e_embd)

        # compute rrwp, and merge
        if self.cfg.rrwp_walk_length > 0:
            # get the connected (idx=1) edge
            _, bidi_exst_e = half_edge_to_full_edge(e_index, e[:, 1])
            _, bidi_e_rrwp = compute_degree_and_rrwp(
                bidi_e_index, bidi_exst_e, n_node, walk_length=self.cfg.rrwp_walk_length
            )
            bidi_e_embd = torch.cat([bidi_e_embd, bidi_e_rrwp], dim=1)
            bidi_e_embd = self.merge_embd_e(bidi_e_embd)

        # merge global features
        timestep_embd = self.embd_timestep(timestep.unsqueeze(1))
        latent_z_embd = self.embd_latent_z(z)
        frag_bag_embd_ = frag_bag_embd.unsqueeze(0).repeat(bs, 1)
        g_embd = torch.cat([frag_bag_embd_, timestep_embd, latent_z_embd], dim=1)
        g_embd = self.merge_embd_g(g_embd)

        # pass through backbone
        if self.cfg.backbone_type in ["gt", "mpnn"]:
            for layer in self.layers:
                h_embd, bidi_e_embd, g_embd = layer(
                    h_embd, bidi_e_index, bidi_e_embd, g_embd, batch
                )
        elif self.cfg.backbone_type in ["gt_digress"]:
            dense_h, dense_h_mask = to_dense_batch(h_embd, batch)
            dense_e = to_dense_adj(bidi_e_index, batch, bidi_e_embd)
            for layer in self.layers:
                dense_h, dense_e, g_embd = layer(dense_h, dense_e, g_embd, dense_h_mask)
            h_embd = dense_h[dense_h_mask]
            unique_graphs, counts = torch.unique_consecutive(batch, return_counts=True)
            batch_starting_indices = torch.cat(
                [
                    torch.tensor([0], device=batch.device),
                    torch.cumsum(counts, dim=0)[:-1],
                ]
            )
            src, tgt = bidi_e_index  # source and target node indices for each edge
            graph_indices = batch[src]
            local_src = src - batch_starting_indices[graph_indices]
            local_tgt = tgt - batch_starting_indices[graph_indices]
            bidi_e_embd = dense_e[graph_indices, local_src, local_tgt]
        else:
            raise NotImplementedError

        # make unidrected graph
        _, e_embd = full_edge_to_half_edge(bidi_e_index, bidi_e_embd)

        # readout edge type
        e_logit = self.fc_readout_e(e_embd)

        # readout latent z
        z = self.fc_readout_latent_z(g_embd)

        return h_embd, e_logit, z
