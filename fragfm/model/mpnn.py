import torch
from torch import nn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as pyg_softmax

from fragfm.model.layer import *


class GlobalMPNNLayer(nn.Module):
    """
    A MPNN layer with global features updates node and edge features
    """

    def __init__(
        self,
        h_dim: int,
        e_dim: int,
        g_dim: int,
        hid_dim: int,
        dropout: float,
        layer_norm: bool,
        activation: str,
        init_method: str,
    ):
        super().__init__()

        if activation == "none":
            pass
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif activation == "softplus":
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

        self.h_dim, self.e_dim, self.hid_dim = h_dim, e_dim, hid_dim
        self.map_g = MLP(dims=[g_dim, hid_dim], init_method=init_method)

        self.fc_m = MLP(
            dims=[h_dim * 2 + e_dim, hid_dim, hid_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            last_activation=activation,
            init_method=init_method,
        )
        self.fc_m_att = MLP(dims=[hid_dim, 1], init_method=init_method)

        self.fc_h = MLP(
            dims=[h_dim + hid_dim, h_dim, h_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            init_method=init_method,
        )
        self.fc_h_att = MLP(dims=[h_dim, 1], init_method=init_method)

        self.fc_e = MLP(
            dims=[e_dim + hid_dim, e_dim, e_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            init_method=init_method,
        )
        self.fc_e_att = MLP(dims=[e_dim, 1], init_method=init_method)

        self.fc_g = MLP(
            dims=[g_dim + h_dim + e_dim, g_dim, g_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            init_method=init_method,
        )

    def forward(self, h, e_index, e, g, batch):
        # map g to hidden dimension
        g_map = self.map_g(g)  # [bs, hid_dim]

        # make message
        m = self.fc_m(
            torch.cat([h[e_index[1]], h[e_index[0]], e], dim=1)
        )  # [n_edge, hid_dim]

        # update e e_{ij} = f(e_{ij}, m_{ij} + g_map)
        e_update = self.fc_e(
            torch.cat([e, m + g_map[batch[e_index[1]]]], dim=1)
        )  # [n_edge, e_dim]
        e_out = self.act(e + e_update)

        # aggr m -> h
        m_att_score = self.fc_m_att(m).squeeze(1)  # [n_edge]
        m_att_prob = pyg_softmax(m_att_score, e_index[1])  # [n_edge]
        m_aggr_to_h = scatter(
            m * m_att_prob.unsqueeze(1),
            e_index[1],
            dim=0,
            dim_size=h.size(0),
            reduce="sum",
        )  # [n_node, hid_dim]

        # update h
        h_update = self.fc_h(
            torch.cat([h, m_aggr_to_h + g_map[batch]], dim=1)
        )  # [n_node, h_dim]
        h_out = self.act(h + h_update)

        # aggr h -> g
        h_att_score = self.fc_h_att(h).squeeze(1)
        h_att_prob = pyg_softmax(h_att_score, batch)
        h_aggr_to_g = scatter(
            h * h_att_prob.unsqueeze(1), batch, dim=0, dim_size=g.size(0), reduce="sum"
        )

        # aggr e -> g
        e_att_score = self.fc_e_att(e).squeeze(1)
        e_att_prob = pyg_softmax(e_att_score, batch[e_index[1]])
        e_aggr_to_g = scatter(
            e * e_att_prob.unsqueeze(1),
            batch[e_index[1]],
            dim=0,
            dim_size=g.size(0),
            reduce="sum",
        )

        # add to g
        g_update = self.fc_g(
            torch.cat([g, h_aggr_to_g, e_aggr_to_g], dim=1)
        )  # [n_node, h_dim]
        g_out = self.act(g + g_update)

        return h_out, e_out, g_out


class GlobalMPNNLayer2(nn.Module):
    """
    A MPNN layer with global features updates node and edge features
    """

    def __init__(
        self,
        h_dim: int,
        e_dim: int,
        g_dim: int,
        hid_dim: int,
        dropout: float,
        layer_norm: bool,
        activation: str,
        init_method: str,
    ):
        super().__init__()

        if activation == "none":
            pass
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif activation == "softplus":
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

        self.h_dim, self.e_dim, self.hid_dim = h_dim, e_dim, hid_dim

        self.fc_m = MLP(
            dims=[h_dim * 2 + e_dim + g_dim, hid_dim, hid_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            last_activation=activation,
            init_method=init_method,
        )
        self.fc_m_att = MLP(
            dims=[hid_dim, 1], last_activation="sigmoid", init_method=init_method
        )

        self.fc_h = MLP(
            dims=[h_dim + hid_dim + g_dim, h_dim, h_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            init_method=init_method,
        )
        self.fc_h_att = MLP(dims=[h_dim, 1], init_method=init_method)

        self.fc_e = MLP(
            dims=[hid_dim, e_dim, e_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            init_method=init_method,
        )

        self.fc_g = MLP(
            dims=[g_dim + h_dim, g_dim, g_dim],
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            init_method=init_method,
        )

    def forward(self, h, e_index, e, g, batch):
        e_batch = batch[e_index[1]]
        # make message
        m_cat = torch.cat([h[e_index[0]], h[e_index[1]], e, g[e_batch]], dim=1)
        m = self.fc_m(m_cat)
        m_att = self.fc_m_att(m)

        # update e
        e_update = self.fc_e(m)
        e_out = self.act(e + e_update)

        # aggregate m
        m_aggr = scatter(m_att * m, e_index[1], dim=0, dim_size=h.size(0), reduce="sum")

        # update h
        h_cat = torch.cat([h, m_aggr, g[batch]], dim=1)
        h_update = self.fc_h(h_cat)
        h_out = self.act(h + h_update)

        # update g
        h_att = self.fc_h_att(h_out)
        h_aggr = scatter(
            h_att * h_out, batch, dim=0, dim_size=batch.max() + 1, reduce="sum"
        )
        g_cat = torch.cat([g, h_aggr], dim=1)
        g_update = self.fc_g(g_cat)
        g_out = self.act(g + g_update)

        return h_out, e_out, g_out
