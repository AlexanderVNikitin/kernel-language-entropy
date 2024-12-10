import os
import wandb
import numpy as np
import collections
import pickle
import json
import pandas as pd
from loguru import logger


DEFAULT_HP = {
    "Weighted Heat": "weighted_heat_t=0.3_kernel_entropy",
    "Weighted Matern": "weighted_matern_kappa=1.0_nu=1.0_kernel_entropy",
    "Semantic Heat": "semantic_kernel_heat_t=0.3_alpha_0.5",
    "Semantic Matern": "semantic_kernel_matern_kappa=1.0_nu=1.0_alpha_0.5",
    "Semantic Prod": "semantic_kernel_prod_matern_kappa=1.0_nu=1.0",
    "Semantic Sum": "semantic_kernel_sum_matern_kappa=1.0_nu=1.0",
    "Full KLE": "full_klu_heat_t=0.3_alpha_0.5",
    "Deberta Weighted": "deberta_full_klu_heat_t=0.3_alpha_0.5",
    "Heat": "heat_t=0.3_kernel_entropy_s",
    "Matern": "matern_kappa=1.0_nu=1.0_kernel_entropy_s",
    "All KLEs": "full_klu_heat_t=0.3_alpha_1.0",
    "Deberta Weighted C": "weighted_deberta_heat_t=0.3_kernel_entropy",
    "u_deg": "weighted_deberta_u_deg",
    "eigv": "weighted_deberta_eigv",
}


def restore_args(api, wandb_id):
    filename = "experiment_details.pkl" 
    folder_path = f'data/wandb/{wandb_id}'
    old_run = api.run(f'an23/semantic_uncertainty/{wandb_id}')
    old_run.file(filename).download(
        replace=True, exist_ok=False, root=folder_path)

    
    with open(os.path.join(folder_path, filename), "rb") as infile:
        old_exp = pickle.load(infile)
    return old_exp["args"]


def kernels_from_results(results):
    """
    returns a dictionary of aggregated kernel names:
    base_name: [raw_name_hp1=v1_...,]
    """
    kernels = collections.defaultdict(list)
    for key in results["uncertainty"].keys():
        #if key.startswith("heat") and not "UNANSWERABLE" in key:
        #    kernels["Heat"].append(key)
        #elif key.startswith("matern") and not "UNANSWERABLE" in key:
        #    kernels["Matern"].append(key)
        if key.startswith("weighted_heat_") and not "UNANSWERABLE" in key:
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
        elif (key.startswith("full_klu_")) and  not "UNANSWERABLE" in key:
            kernels["Full KLE"].append(key)
        elif key.startswith("deberta") and not "UNANSWERABLE" in key:
            kernels["Deberta Weighted"].append(key)
        elif key.startswith("weighted_deberta_") and not "UNANSWERABLE" in key:
            kernels["Deberta Weighted C"].append(key)
        
        if "eigv" in key:
            kernels["eigv"].append(key)
        elif "u_deg" in key:
            kernels["u_deg"].append(key)

        if ("kernel" in key or ("weighted" in key and not "eigv" in key and not "u_deg" in key) or "deberta" in key or "matern" in key or "heat" in key) and not "UNANSWERABLE" in key:
            kernels["All KLEs"].append(key)
    return kernels


def choose_best_hyperparams_from_list(results, list_of_methods):
    """
    Chooses the best hyperparams according to its AUROC
    from raw result. 
    Returns (best_method, best_result)
    """
    results = results["uncertainty"]
    best_auroc, best_method = -np.inf, None
    for method_name in list_of_methods:
        if "AUROC_hp" not in results[method_name]:
            #logger.warning(f"No AUROC on val set for {method_name}")
            continue
        cur_auroc = results[method_name]["AUROC_hp"]["mean"]
        if best_auroc < cur_auroc:
            best_auroc = cur_auroc
            best_method = method_name
    return best_method, best_auroc


def restore_file(wandb_id, filename='wandb-summary.json'):
    files_dir = 'notebooks/restored_files'    
    os.system(f'mkdir -p {files_dir}')

    api = wandb.Api()
    run = api.run(f'semantic_uncertainty/{wandb_id}')

    path = f'{files_dir}/{filename}'
    os.system(f'rm -rf {path}')
    run.file(filename).download(root=files_dir, replace=True, exist_ok=False)
    with open(path, 'r') as f:
        out = json.load(f)
    return out


def df_failed_pik(df):
    if "p_ik" not in df.method.unique():
        logger.error("No p_ik, defaulting to 0")
        metrics = df.metric.unique()
        new_data = {
            'method': ["p_ik"] * len(metrics),
            'metric': metrics,
            'means': [0] * len(metrics),
            'err': [0]* len(metrics)
        }
        new_df = pd.DataFrame(new_data)
        return pd.concat([df, new_df], ignore_index=True)
    else:
        return df


def get_uncertainty_df(metrics, select_best_hp=True, verbose=False):
    data = []
    for method in metrics['uncertainty']:
        for metric in metrics['uncertainty'][method]:
            mean = metrics['uncertainty'][method][metric]['mean']
            if isinstance(metrics['uncertainty'][method][metric]["bootstrap"], float) and np.isnan(metrics['uncertainty'][method][metric]["bootstrap"]):
                err = np.nan
            else:
                err = metrics['uncertainty'][method][metric]["bootstrap"]['std_err']
            data.append([method, metric, mean, err])
    df = pd.DataFrame(data, columns=['method', 'metric', 'means', 'err'])
    #main_methods = ['semantic_entropy', 'cluster_assignment_entropy', 'regular_entropy', 'p_false', 'p_ik',
    #                "heat_kernel_entropy", "matern_kernel_entropy", "weighted_heat_kernel_entropy", "weighted_matern_kernel_entropy"]
    kernels = kernels_from_results(metrics)
    main_methods = ['semantic_entropy', 'cluster_assignment_entropy', 'regular_entropy', 'p_false', 'p_ik']
    main_names = ['Semantic entropy', 'Discrete Semantic Entropy', 'Naive Entropy', 'p(True)', 'Embedding Regression']
    for kernel_name, hp_names in kernels.items():
        if select_best_hp:
            best_hp, metric = choose_best_hyperparams_from_list(
                metrics, hp_names)
        else:
            best_hp, metric = DEFAULT_HP[kernel_name], "default"
        if best_hp is None:
            if verbose:
                logger.info(f"Using default hyperparams")
            best_hp = DEFAULT_HP[kernel_name]
        if verbose:
            logger.info(f"Best Hyperparams for {kernel_name} are: {best_hp} with {metric}")
        rows_to_duplicate = df[df['method'] == best_hp].copy()
        rows_to_duplicate['method'] = kernel_name
        df = pd.concat([df, rows_to_duplicate], ignore_index=True)
        if len(rows_to_duplicate):
            main_names.append(kernel_name)
            main_methods.append(kernel_name)
    df = df_failed_pik(df)
    df = df.set_index('method').loc[main_methods].reset_index()
    conversion = dict(zip(main_methods, main_names))
    df['method'] = df.method.map(lambda x: conversion[x])
    return df
