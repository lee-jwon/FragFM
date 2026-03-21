import torch
from easydict import EasyDict
from torch_geometric.utils import to_dense_adj, to_dense_batch

from fragfm.model.flow import CoarseGraphPropagate, FragToVect
from fragfm.model.layer import *


class FragToVectReadout(FragToVect):
    def __init__(self, cfg: EasyDict):
        super().__init__(cfg)
        self.disc_readout_prop = MLP(
            [cfg.embd_h_dim, cfg.embd_h_dim, cfg.embd_h_dim, 2],
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
        prop = self.disc_readout_prop(z)
        return z, prop.squeeze(1)


class CoarseGraphReadout(CoarseGraphPropagate):
    def __init__(self, cfg: EasyDict):
        super().__init__(cfg)
        self.cfg = cfg

        # added only for discriminator
        self.disc_readout_prop = MLP(
            [
                cfg.embd_h_dim,
                cfg.embd_h_dim,
                2,
            ],
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

        """# readout edge type
        e_logit = self.fc_readout_e(e_embd)

        # readout latent z
        z = self.fc_readout_latent_z(g_embd)"""

        # readout all
        prop = self.disc_readout_prop(g_embd)
        return prop
