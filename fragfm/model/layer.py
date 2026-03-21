import math

import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch_geometric.utils import scatter, to_dense_adj


def half_edge_to_full_edge(edge_index, edge_attr):
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return edge_index, edge_attr


def full_edge_to_half_edge(edge_index, edge_attr):
    """
    this should be use in caution !!!
    """
    n_edge = edge_attr.size(0)
    edge_index = edge_index[:, : n_edge // 2]
    edge_attr = edge_attr[: n_edge // 2, :] + edge_attr[n_edge // 2 :, :]
    return edge_index, edge_attr


class SinusoidalEncodingLayer(nn.Module):
    def __init__(self, max_len, embedding_dim, use_fc=False):
        super().__init__()
        self.embedding_dim = embedding_dim

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )
        sinusoid = torch.zeros(max_len, embedding_dim)
        sinusoid[:, 0::2] = torch.sin(position * div_term)
        sinusoid[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("sinusoid", sinusoid)

        self.use_fc = use_fc
        if use_fc:
            self.fc = MLP([embedding_dim, embedding_dim], init_method="he")

    def forward(self, x):
        if self.use_fc:
            return self.fc(self.sinusoid[x])
        else:
            return self.sinusoid[x]


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        dropout=0.0,
        layer_norm=False,
        activation="relu",
        last_activation="none",
        init_method="default",
        last_layer_xavier_small=False,
        bias=True,
    ):
        super().__init__()
        assert len(dims) > 1  # more than two dims (in out)
        assert activation in [
            "none",
            "relu",
            "silu",
            "leaky_relu",
            "softplus",
        ]
        assert last_activation in [
            "none",
            "relu",
            "silu",
            "leaky_relu",
            "softplus",
            "sigmoid",
            "tanh",
        ]
        assert init_method in [
            "default",
            "xavier",
            "he",
        ]

        n_layer = len(dims)
        layers = []
        for i in range(n_layer - 1):  # 0, 1, ..., n_layer - 2
            in_dim, out_dim = dims[i], dims[i + 1]

            # Parameter initialization
            if init_method == "default":
                init_func = None  # Use PyTorch default initialization
            elif init_method == "xavier":
                init_func = nn.init.xavier_uniform_
            elif init_method == "he":
                init_func = nn.init.kaiming_uniform_
            else:
                raise ValueError
            bias_init_func = nn.init.zeros_

            # Linear layer
            linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
            if init_func is not None:
                init_func(linear_layer.weight)
                if bias:
                    bias_init_func(linear_layer.bias)
            layers.append(linear_layer)

            if i < n_layer - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

                if activation == "none":
                    pass
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                elif activation == "softplus":
                    layers.append(nn.Softplus())
                else:
                    raise NotImplementedError

        if last_layer_xavier_small:
            torch.nn.init.xavier_uniform_(layers[-1].weight, gain=0.001)

        if last_activation == "none":
            pass
        elif last_activation == "relu":
            layers.append(nn.ReLU())
        elif last_activation == "silu":
            layers.append(nn.SiLU())
        elif last_activation == "leaky_relu":
            layers.append(nn.LeakyReLU())
        elif last_activation == "softplus":
            layers.append(nn.Softplus())
        elif last_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise NotImplementedError

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NodeToGlobal(nn.Module):
    def __init__(self, h_dim, g_dim, init_method):
        super().__init__()

        # self.fc = MLP([h_dim * 4, g_dim], init_method=init_method)
        self.fc = Linear(h_dim * 4, g_dim)

    def forward(self, h, batch):
        h1 = scatter(h, batch, dim=0, dim_size=batch.max() + 1, reduce="mean")
        h2 = scatter(h, batch, dim=0, dim_size=batch.max() + 1, reduce="min")
        h3 = scatter(h, batch, dim=0, dim_size=batch.max() + 1, reduce="max")
        h4 = scatter_std(h, batch, dim=0, dim_size=batch.max() + 1)
        """h_diff_sq = (h - h1[batch]) ** 2
        h_var = scatter(
            h_diff_sq, batch, dim=0, dim_size=batch.max() + 1, reduce="mean"
        )
        h4 = torch.sqrt(h_var)"""
        h = torch.cat([h1, h2, h3, h4], dim=1)
        return self.fc(h)


class EdgeToGlobal(nn.Module):
    def __init__(self, e_dim, g_dim, init_method):
        super().__init__()
        print("X" * 1000)

        # self.fc = MLP([e_dim * 4, g_dim], init_method=init_method)
        self.fc = Linear(e_dim * 4, g_dim)

    def forward(self, e_index, e, batch):
        e_batch = batch[e_index[1]]
        e1 = scatter(e, e_batch, dim=0, dim_size=batch.max() + 1, reduce="mean")
        e2 = scatter(e, e_batch, dim=0, dim_size=batch.max() + 1, reduce="min")
        e3 = scatter(e, e_batch, dim=0, dim_size=batch.max() + 1, reduce="max")
        e4 = scatter_std(e, e_batch, dim=0, dim_size=batch.max() + 1)
        """e_diff_sq = (e - e1[e_batch]) ** 2
        e_var = scatter(
            e_diff_sq, e_batch, dim=0, dim_size=batch.max() + 1, reduce="mean"
        )
        e4 = torch.sqrt(e_var)"""
        e = torch.cat([e1, e2, e3, e4], dim=1)
        return self.fc(e)


def compute_degree_and_rrwp(e_index, e, n_node, walk_length=8):
    """
    e_index would be fully connected, anyway...
    e would be a 1D tensor of 0 (disconnected), 1 (connected)
    """
    assert len(e.size()) == 1, "size error"
    with torch.no_grad():
        e_mask = e.bool()
        exst_e_index = e_index[:, e_mask]
        adj = to_dense_adj(exst_e_index, max_num_nodes=n_node)[0]
        d = adj.sum(dim=1)
        d_inv = torch.diag(1.0 / d)
        d_inv[d_inv == float("inf")] = 0
        m = d_inv @ adj

        rrwps = [m]
        for i in range(walk_length - 1):
            rrwps.append(rrwps[-1] @ m)
        rrwp = torch.stack(rrwps, dim=2)

        e_rrwp = rrwp[e_index[0], e_index[1]]
    return d, e_rrwp
