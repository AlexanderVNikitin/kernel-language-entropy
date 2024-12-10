import os
import pickle
import logging
from collections import defaultdict

import numpy as np
import wandb
import torch
import networkx as nx
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.utils import openai as oai
from uncertainty.utils import utils


from uncertainty.uncertainty_measures.semantic_entropy import *


def get_entailment_graph(strings_list, model, is_weighted=False, example=None, weight_strategy="manual"):
    """
    Get graph of entailment
    """
    def get_edge(text1, text2, is_weighted=False, example=None):
        implication_1, prob_impl1 = model.check_implication(text1, text2, example=example)
        implication_2, prob_impl2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2])
        weight = int(implication_1 == 2) + int(implication_2 == 2) + 0.5 * int(implication_1 == 1) + 0.5 * int(implication_2 == 1)
        if is_weighted:
            if weight_strategy == "manual":
                return weight
            elif weight_strategy == "deberta":
                return prob_impl1 + prob_impl2
            else:
                raise ValueError(f"Unknown weight strategy {weight_strategy}")
        return weight >= 1.5

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    nodes = range(len(strings_list))
    edges = []
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                edge = get_edge(string1, strings_list[j], example=example, is_weighted=is_weighted)
                if is_weighted:
                    if edge:
                        edges.append((i, j, edge))
                else:
                    edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    return G


def get_semantic_ids_graph(strings_list, model, semantic_ids, ordered_ids, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    def are_similar(text1, text2):

        implication_1, prob = model.check_implication(text1, text2, example=example)
        implication_2, prob = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        return (implication_1 == 2) + (implication_1 == 1) * 0.5 +\
               (implication_2 == 2) + (implication_2 == 1) * 0.5

    # Initialise all ids with -1.
    nodes = ordered_ids
    weights = defaultdict(list) # (i, j) -> weight
    for i, string1 in enumerate(strings_list):
        node_i = semantic_ids[i]
        for j in range(i + 1, len(strings_list)):
            node_j = semantic_ids[j]
            edge_weight = are_similar(string1, strings_list[j])
            if edge_weight > 0:
                weights[(node_i, node_j)].append(edge_weight)
    for k, v in weights.items():
        weights[k] = np.sum(v)
    assert -1 not in semantic_ids
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(i, j, w) for (i, j), w in weights.items()])
    return G
