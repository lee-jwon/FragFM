from copy import deepcopy

import numpy as np


def slice_array(x: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    x = x[s1]  # [a1, N, *]
    x = np.swapaxes(x, 0, 1)  # [N, a1, *]
    x = x[s2]  # [a2, a1, *]
    x = np.swapaxes(x, 0, 1)  # [a1, a2, *]
    return x


def get_independent_nodes_from_adj(adj: np.ndarray):
    adj = np.array(adj)
    n = adj.shape[0]
    vis = [False] * n
    ind_nodes = []

    def dfs(node, comp):
        vis[node] = True
        comp.append(node)
        for neighbor, is_connected in enumerate(adj[node]):
            if is_connected and not vis[neighbor]:
                dfs(neighbor, comp)

    for i in range(n):
        if not vis[i]:
            comp = []
            dfs(i, comp)
            ind_nodes.append(comp)
    return ind_nodes


def e_index_e_to_adje(e_index: np.ndarray, e: np.ndarray, n: int) -> np.ndarray:
    import numpy as np

    num_nodes = np.max(e_index) + 1
    if e.ndim > 1:
        adj_shape = (n, n) + e.shape[1:]
    else:
        adj_shape = (n, n)
    adj_matrix = np.zeros(adj_shape, dtype=e.dtype)

    for idx in range(e_index.shape[1]):
        src = e_index[0, idx]
        tgt = e_index[1, idx]
        adj_matrix[src, tgt] = e[idx]
        adj_matrix[tgt, src] = e[idx]
    return adj_matrix


def mask_pairs_from_adje(adje: np.ndarray, pair: np.ndarray) -> np.ndarray:  # [2, *]
    adje = adje.copy()
    for src, tgt in pair.T:
        adje[src, tgt] = 0
        adje[tgt, src] = 0
    return adje


def sparse_edge_to_fully_connected_edge(
    e_index: np.ndarray, e: np.ndarray, n_node: int, pad_val: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.triu_indices(n_node, k=1)  # Include self-loops
    source_nodes = indices[0]
    target_nodes = indices[1]

    new_source_nodes = []
    new_target_nodes = []
    new_e = []

    edge_dict = {}
    num_edges = e_index.shape[1]
    for i in range(num_edges):
        src = e_index[0, i]
        tgt = e_index[1, i]
        edge_key = tuple(sorted((src, tgt)))
        edge_dict[edge_key] = e[i]

    feature_shape = e.shape[1:] if e.ndim > 1 else ()

    for src, tgt in zip(source_nodes, target_nodes):
        edge_key = tuple(sorted((src, tgt)))
        if edge_key in edge_dict:
            edge_feature = edge_dict[edge_key]
        else:
            # Assign default features (e.g., pad_val)
            edge_feature = np.full(feature_shape, pad_val)

        new_source_nodes.append(src)
        new_target_nodes.append(tgt)
        new_e.append(edge_feature)

    new_e_index = np.vstack((new_source_nodes, new_target_nodes))
    new_e = np.array(new_e)

    return new_e_index.astype(int), new_e.astype(int)


def adje_to_sparse_edge(adje: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = adje.shape[0]
    edge_list = []
    edge_vals = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.any(adje[i, j] != 0):
                edge_list.append([i, j])
                edge_vals.append(adje[i, j])
    edge_list = np.array(edge_list).T
    edge_vals = np.array(edge_vals)
    if len(edge_list) == 0:
        edge_list = np.empty((2, 0)).astype(int)
    return edge_list, edge_vals
