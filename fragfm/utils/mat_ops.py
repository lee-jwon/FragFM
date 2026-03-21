import networkx as nx
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torch_geometric.utils import scatter


def max_weight_matching_mask(A: torch.Tensor) -> torch.Tensor:
    """
    Conduct Blossom algorithm
    Given an (N x N) PyTorch tensor A, find a maximum weight matching
    and return an (N x N) boolean mask where True indicates the matched edges.

    - We assume A[i,j] = A[j,i] for an undirected graph (not strictly required by networkx, but typical).
    - If (i, j) is in the matching, we set mask[i,j] = mask[j,i] = True.
    - All other entries are False.
    """
    # Convert to NumPy for networkx
    A_np = A.detach().cpu().numpy()
    N = A_np.shape[0]

    # Build graph
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=A_np[i, j])

    # Maximum weight matching
    matching_set = nx.max_weight_matching(G, maxcardinality=True)

    # Create an NxN boolean mask
    mask = torch.zeros((N, N), dtype=torch.bool)

    # Fill in True for the matched edges
    for i, j in matching_set:
        mask[i, j] = True
        mask[j, i] = True

    return mask


def sample_from_prob(x: torch.Tensor, return_onehot: bool = False) -> torch.Tensor:
    n = x.size(1)
    x = x / x.sum(keepdim=True, dim=1)
    c = Categorical(x).sample()
    if return_onehot:
        return F.one_hot(c, num_classes=n).float()
    return c


def prob_to_argmax_onehot(prob_matrix: torch.Tensor) -> torch.Tensor:
    max_indices = torch.argmax(prob_matrix, dim=1)
    one_hot = torch.zeros_like(prob_matrix)
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
    return one_hot
