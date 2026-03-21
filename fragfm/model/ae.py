import torch
from easydict import EasyDict
from torch import nn
from torch_geometric.utils import scatter

from fragfm.model.layer import *
from fragfm.model.mpnn import GlobalMPNNLayer


class FragJunctionAE(nn.Module):
    def __init__(self, cfg: EasyDict):
        super().__init__()
        self.cfg = cfg

        if cfg.embd_method == "sinusoidal":
            self.encoder_embd_h = SinusoidalEncodingLayer(
                300, cfg.embd_h_dim // 4, use_fc=False
            )
            self.encoder_embd_h_junction_count = nn.Embedding(5, cfg.embd_h_dim)
            self.encoder_embd_h_in_frag_label = SinusoidalEncodingLayer(
                300, cfg.embd_h_dim // 4, use_fc=False
            )
            self.encoder_embd_h_aux_frag_label = SinusoidalEncodingLayer(
                30, cfg.embd_h_dim // 4 + cfg.embd_h_dim % 4, use_fc=False
            )
            self.decoder_embd_h = SinusoidalEncodingLayer(
                300, cfg.embd_h_dim // 4, use_fc=False
            )
            self.decoder_embd_h_junction_count = nn.Embedding(5, cfg.embd_h_dim // 4)
            self.decoder_embd_h_in_frag_label = SinusoidalEncodingLayer(
                300, cfg.embd_h_dim // 4, use_fc=False
            )
            self.decoder_embd_h_aux_frag_label = SinusoidalEncodingLayer(
                30, cfg.embd_h_dim // 4 + cfg.embd_h_dim % 4, use_fc=False
            )
        elif cfg.embd_method == "nn":
            self.encoder_embd_h = nn.Embedding(300, cfg.embd_h_dim // 4)
            self.encoder_embd_h_junction_count = nn.Embedding(5, cfg.embd_h_dim // 4)
            self.encoder_embd_h_in_frag_label = nn.Embedding(300, cfg.embd_h_dim // 4)
            self.encoder_embd_h_aux_frag_label = nn.Embedding(
                30, cfg.embd_h_dim // 4 + cfg.embd_h_dim % 4
            )
            self.decoder_embd_h = nn.Embedding(300, cfg.embd_h_dim // 4)
            self.decoder_embd_h_junction_count = nn.Embedding(5, cfg.embd_h_dim // 4)
            self.decoder_embd_h_in_frag_label = nn.Embedding(300, cfg.embd_h_dim // 4)
            self.decoder_embd_h_aux_frag_label = nn.Embedding(
                30, cfg.embd_h_dim // 4 + cfg.embd_h_dim % 4
            )
        else:
            raise NotImplementedError

        self.encoder_embd_e = nn.Embedding(5, cfg.embd_e_dim)
        self.decoder_embd_e = nn.Embedding(5, cfg.embd_e_dim)

        encoder_layers = []
        for _ in range(cfg.backbone_n_encoder_layer):
            encoder_layers.append(
                GlobalMPNNLayer(
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
        self.encoder_layers = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for _ in range(cfg.backbone_n_decoder_layer):
            decoder_layers.append(
                GlobalMPNNLayer(
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
        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.fc_z_mu = MLP(
            [cfg.embd_h_dim, cfg.embd_h_dim, cfg.latent_z_dim],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )
        self.fc_z_logvar = MLP(
            [cfg.embd_h_dim, cfg.embd_h_dim, cfg.latent_z_dim],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

        self.embd_latent_z = MLP(
            [cfg.latent_z_dim, cfg.embd_h_dim, cfg.embd_h_dim],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

        self.readout_e = MLP(
            [cfg.embd_e_dim, cfg.embd_e_dim, 1],
            dropout=cfg.fc_dropout,
            layer_norm=cfg.fc_layer_norm,
            activation=cfg.fc_activation,
            init_method=cfg.fc_init_method,
        )

    def forward(
        self,
        h,
        h_junction_count,
        h_in_frag_label,
        h_aux_frag_label,
        e_index,
        e,
        decomp_e_index,
        decomp_e,
        ae_to_pred_index,
        batch,
    ):
        model_out = {}
        z_mu, z_logvar = self.encode(
            h, h_junction_count, h_in_frag_label, h_aux_frag_label, e_index, e, batch
        )
        z = self.reparameterize(z_mu, z_logvar)
        e = self.decode(
            z,
            h,
            h_junction_count,
            h_in_frag_label,
            h_aux_frag_label,
            decomp_e_index,
            decomp_e,
            ae_to_pred_index,
            batch,
        )
        model_out["z_mu"] = z_mu
        model_out["z_logvar"] = z_logvar
        model_out["z"] = z
        model_out["e"] = e
        return model_out

    def encode(
        self, h, h_junction_count, h_in_frag_label, h_aux_frag_label, e_index, e, batch
    ):
        h1 = self.encoder_embd_h(h)
        h2 = self.encoder_embd_h_junction_count(h_junction_count)
        h3 = self.encoder_embd_h_in_frag_label(h_in_frag_label.clamp(max=299))
        h4 = self.encoder_embd_h_aux_frag_label(h_aux_frag_label.clamp(max=29))
        h_embd = torch.cat([h1, h2, h3, h4], dim=1)

        e_embd = self.encoder_embd_e(e)

        g_embd = scatter(h_embd, batch, dim=0, dim_size=batch.max() + 1, reduce="mean")

        bidi_e_index, e_embd = half_edge_to_full_edge(e_index, e_embd)
        for layer in self.encoder_layers:
            h_embd, e_embd, g_embd = layer(h_embd, bidi_e_index, e_embd, g_embd, batch)

        z_mu = self.fc_z_mu(g_embd)
        z_logvar = self.fc_z_logvar(g_embd)
        return z_mu, z_logvar

    def decode(
        self,
        z,
        h,
        h_junction_count,
        h_in_frag_label,
        h_aux_frag_label,
        decomp_e_index,
        decomp_e,
        ae_to_pred_index,
        batch,
    ):
        assert torch.all(decomp_e != 0)

        e_mask_type = torch.zeros([ae_to_pred_index.size(1)]).to(h.device).int()

        h1 = self.decoder_embd_h(h)
        h2 = self.decoder_embd_h_junction_count(h_junction_count)
        h3 = self.decoder_embd_h_in_frag_label(h_in_frag_label.clamp(max=299))
        h4 = self.decoder_embd_h_aux_frag_label(h_aux_frag_label.clamp(max=29))
        h_embd = torch.cat([h1, h2, h3, h4], dim=1)

        if self.training:
            mask = torch.rand(h.size(0)) > self.cfg.decoder_drop_node
            h_embd = h_embd.clone()
            h_embd[~mask, :] = 0.0

        all_e_index = torch.cat([ae_to_pred_index, decomp_e_index], dim=1)  # [2, *]
        all_e = torch.cat([e_mask_type, decomp_e])  # 1~4 when real bond, 0 when to pred

        e_embd = self.decoder_embd_e(all_e)
        g_embd = self.embd_latent_z(z)

        bidi_all_e_index, e_embd = half_edge_to_full_edge(all_e_index, e_embd)
        for layer in self.decoder_layers:
            h_embd, e_embd, g_embd = layer(
                h_embd, bidi_all_e_index, e_embd, g_embd, batch
            )
        _, e_embd = full_edge_to_half_edge(bidi_all_e_index, e_embd)

        e_embd = e_embd[all_e == 0]  # leave only to_preds
        e_out = self.readout_e(e_embd)
        return e_out.squeeze(1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal tensor
        return mu + eps * std

    def get_min_max_transform(self, data_loader):
        """
        Computes the maximum and minimum value of z, among the data_loader.
        The min and max are computed as
        """
        from tqdm import tqdm

        out = {}
        zs_list = []
        n_zs = 0
        for graph, _ in tqdm(data_loader):
            graph.to("cuda")
            with torch.no_grad():
                z, _ = self.encode(
                    graph.h,
                    graph.h_junction_count,
                    graph.h_in_frag_label,
                    graph.h_aux_frag_label,
                    graph.e_index,
                    graph.e,
                    graph.batch,
                )
            zs_list.append(z)
            n_zs += z.size(0)
        zs = torch.cat(zs_list, dim=0)  # [N, z_dim]
        min_zs = torch.min(zs, dim=0)[0]
        max_zs = torch.max(zs, dim=0)[0]
        out["min"], out["max"] = min_zs.cpu().detach(), max_zs.cpu().detach()
        return out
