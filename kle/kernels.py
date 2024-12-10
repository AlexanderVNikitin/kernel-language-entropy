import networkx as nx
import torch
import scipy
import numpy as np
from numpy.linalg import matrix_power as mp
from scipy.linalg import fractional_matrix_power as fmp


def get_laplacian(G, norm_lapl):
    if isinstance(G, nx.DiGraph):
        L = nx.directed_laplacian_matrix(G)
    elif norm_lapl:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    return L


def heat_kernel(G: nx.Graph, t: float = 0.4, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    return scipy.linalg.expm(-t * L)


def matern_kernel(G: nx.Graph, kappa: float = 1, nu=1, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    I = np.eye(L.shape[0])
    #return fmp(nu * I + L, -alpha / 2) @ fmp(nu * I + L.T, -alpha / 2)
    return fmp((2 * nu / kappa**2) * I + L, -nu)
