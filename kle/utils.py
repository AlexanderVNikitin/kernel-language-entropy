import os
import random
import wandb
import torch
import tqdm
import datasets
import functools
import numpy as np
import pickle
import io
import collections


def fix_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_hf_caches(ds_cache, eval_cache):
    os.environ["HF_DATASETS_CACHE"] = ds_cache
    os.environ["HF_EVALUATE_CACHE"] = eval_cache


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def unpickle(infile, device="cpu"):
    if device == "cpu":
        return CPU_Unpickler(infile).load()
    else:
        return pickle.load(infile)
    

def kernels_from_results(results):
    kernels = collections.defaultdict(list)
    for key in results["uncertainty"].keys():
        if key.startswith("heat") and not "UNANSWERABLE" in key:
            kernels["Heat"].append(key)
        elif key.startswith("matern") and not "UNANSWERABLE" in key:
            kernels["Matern"].append(key)
        elif key.startswith("weighted_heat_") and not "UNANSWERABLE" in key:
            kernels["Weighted Heat"].append(key)
        elif key.startswith("weighted_matern_") and not "UNANSWERABLE" in key:
            kernels["Weighted Matern"].append(key)
        elif key.startswith("weighted_matern_") and not "UNANSWERABLE" in key:
            kernels["Weighted Matern"].append(key)
        elif key.startswith("semantic_kernel_heat") and not "UNANSWERABLE" in key:
            kernels["Semantic Heat"].append(key)
        elif key.startswith("semantic_kernel_matern") and not "UNANSWERABLE" in key:
            kernels["Semantic Matern"].append(key)
        elif key.startswith("semantic_kernel_prod") and not "UNANSWERABLE" in key:
            kernels["Semantic Prod"].append(key)
        elif key.startswith("semantic_kernel_sum") and not "UNANSWERABLE" in key:
            kernels["Semantic Sum"].append(key)
    return kernels


def choose_best_hyperparams(results, base_name="semantic_kernel_heat", method_names=["{base_name}_alpha_0.4"]):
    results = results["uncertainty"]
    best_auroc = -np.inf
    for method_name in method_names:
        method_name = method_name.format(base_name=base_name)
        cur_auroc = results[method_name]["AUROC_hp"]["mean"]
        if best_auroc < cur_auroc:
            best_auroc = cur_auroc
            best_method = method_name
    return best_method, best_auroc


def choose_best_hyperparams_from_list(results, list_of_methods):
    results = results["uncertainty"]
    best_auroc = -np.inf
    for method_name in list_of_methods:
        cur_auroc = results[method_name]["AUROC_hp"]["mean"]
        if best_auroc < cur_auroc:
            best_auroc = cur_auroc
            best_method = method_name
    return best_method, best_auroc


def degree_matrix(G):
    # Get the degree of each node
    degrees = dict(G.degree())
    # Create a diagonal matrix with the degrees
    D = np.diag([degrees[node] for node in G.nodes()])
    return D
