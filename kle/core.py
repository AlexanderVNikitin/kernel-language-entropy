import numpy as np
import torch
import kle
import networkx as nx


EPS = 1e-12


def find_cliques_directed(G):
    G1 = nx.Graph()
    for u,v in G.edges():
        if u in G[v]:
            G1.add_edge(u,v)
    return nx.find_cliques(G1)


def contract_cliques(G, cliques):
    used = [False] * len(G.nodes)
    for clique in cliques:
        first_node = None
        for node in clique:
            if used[node]:
                continue
            if first_node is None:
                first_node = node
            else:
                G = nx.contracted_nodes(G, first_node, node)
                used[first_node] = True
                used[node] = True
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def normalize_kernel(K):
    diagonal_values = np.sqrt(np.diag(K)) + EPS
    normalized_kernel = K / np.outer(diagonal_values, diagonal_values)
    return normalized_kernel


def scale_entropy(entropy, n_classes):
    max_entropy = -np.log(1.0 / n_classes)  # For a discrete distribution with num_classes
    scaled_entropy = entropy / max_entropy
    return scaled_entropy


def vn_entropy(K, normalize=True, scale=True, jitter=0):
    if normalize:
        K = normalize_kernel(K) / K.shape[0]
    result = 0
    eigvs = np.linalg.eig(K + jitter * np.eye(K.shape[0])).eigenvalues.astype(np.float64)
    for e in eigvs:
        if np.abs(e) > 1e-8:
            result -= e * np.log(e)
    if scale:
        result = scale_entropy(result, K.shape[0])
    return np.float64(result)


def contract_cliques_impl(G, cliques):
    used = [False] * len(G.nodes)
    for clique in cliques:
        first_node = None
        for node in clique:
            if used[node]:
                continue
            if first_node is None:
                first_node = node
            else:
                G = nx.contracted_nodes(G, first_node, node)
                used[first_node] = True
                used[node] = True
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
