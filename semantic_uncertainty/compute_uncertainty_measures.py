"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import networkx as nx
import kle.utils
import wandb
import torch

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures.kernel_uncertainty import get_entailment_graph
from uncertainty.uncertainty_measures.kernel_uncertainty import get_semantic_ids_graph
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils

import kle


utils.setup_logger()

# bug in wandb
#os.environ["WANDB__SERVICE_WAIT"] = "300"

EXP_DETAILS = 'experiment_details.pkl'
ALPHAS_RANGE = np.arange(0, 1.01, 0.1)
HEAT_T_RANGE = np.arange(0.1, 0.71, 0.1)
MATERN_KAPPA_RANGE = [1.0, 2.0, 3.0]
MATERN_NU_RANGE = [1.0, 2.0, 3.0]


def get_from_sem_to_sentence_id(ordered_ids):
    from_sem_to_sentence_id = defaultdict(list)
    for i, el in enumerate(ordered_ids):
        from_sem_to_sentence_id[el].append(i)
    return from_sem_to_sentence_id


def reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    new_graph = nx.Graph()
    for sem_id in ordered_sem_ids:
        for sent_id in from_sem_to_sentence_id[sem_id]:
            new_graph.add_node(sent_id)
    
    new_graph.add_edges_from(graph.edges)
    return new_graph


def get_kernels(graph):
    kernels = {}
    for t in HEAT_T_RANGE:
        kernels[f"heat_t={t:.2}"] = kle.kernels.heat_kernel(graph, t=t)
        kernels[f"heatn_t={t:.2}"] = kle.kernels.heat_kernel(graph, t=t, norm_lapl=True)

    for kappa in MATERN_KAPPA_RANGE:
        for nu in MATERN_NU_RANGE:
            kernels[f"matern_kappa={kappa:.2}_nu={nu:.2}"] = kle.kernels.matern_kernel(graph, kappa=kappa, nu=nu)
            kernels[f"maternn_kappa={kappa:.2}_nu={nu:.2}"] = kle.kernels.matern_kernel(graph, kappa=kappa, nu=nu, norm_lapl=True)

    return kernels


def all_graph_entropies(graph):
    kernels = get_kernels(graph)
    results = []
    for kernel_name, kernel in kernels.items():
        for scale in [True, False]:
            kernel_entropy = kle.core.vn_entropy(kernel, scale=scale)
            postfix = "_s" if scale else ""
            results.append((f'{kernel_name}_kernel_entropy{postfix}', kernel_entropy))
    return results


def get_block_diagonal_sem_kernel(log_likelihoods_per_sem_id, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    blocks = []
    for i, sem_id in enumerate(ordered_sem_ids):
        block_size = len(from_sem_to_sentence_id[sem_id])
        blocks.append(torch.exp(torch.tensor(log_likelihoods_per_sem_id[sem_id])) * torch.ones((block_size, block_size)) / block_size)
    return torch.block_diag(*blocks)  


def full_sem_unc_plus_klu(graph, log_likelihoods_per_sem_id, semantic_ids, ordered_sem_ids):
    graph = reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids)
    block_diag_sem_kernel = get_block_diagonal_sem_kernel(
        log_likelihoods_per_sem_id=log_likelihoods_per_sem_id,
        semantic_ids=semantic_ids, ordered_sem_ids=ordered_sem_ids)

    alphas = ALPHAS_RANGE
    results = []
    kernels = get_kernels(graph)
    for kernel_name, kernel in kernels.items():
        for alpha in alphas:
            kernel = kle.core.normalize_kernel(kernel) / kernel.shape[0]
            avg_kernel = alpha * torch.tensor(kernel) + (1 - alpha) * block_diag_sem_kernel
            avg_kernel = avg_kernel.numpy()
            success = False
            for jitter in [0, 1e-16, 1e-12]:
                try:
                    results.append(
                        (f"full_klu_{kernel_name}_alpha_{alpha:.2}", 
                        kle.core.vn_entropy(
                            avg_kernel, normalize=False,
                            scale=False, jitter=jitter
                    )))
                    success = True
                    if jitter > 0:
                        logging.warn(f"Had to use jitter for numerical stability: {jitter}")
                    break
                except:
                    continue
            if not success:
                raise ValueError(f"Unable to calculate VNE for kernel {avg_kernel}")
    return results


def all_semantic_entropies(semantic_graph, log_likelihoods_per_sem_id):
    sem_entropies = torch.diag(torch.exp(torch.tensor(log_likelihoods_per_sem_id)))
    alphas = ALPHAS_RANGE
    results = []
    kernels = get_kernels(semantic_graph)
    for kernel_name, kernel in kernels.items():
        for alpha in alphas:
            kernel = kle.core.normalize_kernel(kernel) / kernel.shape[0]
            avg_kernel = alpha * torch.tensor(kernel) + (1 - alpha) * sem_entropies
            avg_kernel = avg_kernel.numpy()
            results.append(
                (f"semantic_kernel_{kernel_name}_alpha_{alpha:.2}", 
                kle.core.vn_entropy(avg_kernel, normalize=False, scale=False
            )))
    return results


def all_semantic_entropies_diag(semantic_graph, log_likelihoods_per_sem_id):
    sem_entropies = torch.exp(torch.tensor(log_likelihoods_per_sem_id))
    results = []
    kernels = get_kernels(semantic_graph)

    for kernel_name, kernel in kernels.items():
        kernel = kle.core.normalize_kernel(kernel) / kernel.shape[0]
        kernel_prod = torch.tensor(kernel) * sem_entropies
        kernel_sum = torch.tensor(kernel) + sem_entropies
        results.append(
            (f"semantic_kernel_prod_{kernel_name}", 
            kle.core.vn_entropy(kernel_prod, normalize=True, scale=False
        )))
        results.append(
            (f"semantic_kernel_sum_{kernel_name}", 
            kle.core.vn_entropy(kernel_sum, normalize=True, scale=False
        )))
    return results


def main(args):
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    if args.assign_new_wandb_id or args.skip_generation:
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
        if args.assign_new_wandb_id:
            logging.info('Assign new wandb_id.')
            run = wandb.init(
                entity=args.entity,
                project=project,
                dir=wandb_dir,
                notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
                # For convenience, keep any 'generate_answers' configs from old run,
                # but overwrite the rest!
                # NOTE: This means any special configs affecting this script must be
                # called again when calling this script!
                config={**old_run.config, **args.__dict__},
                tags=["eval_only", args.experiment_lot, 
                      f"entailment={args.entailment_model}",
                      f"metric={args.metric}-{args.metric_model}"],
                resume="allow"
            )
        else:
            run = wandb.init(
                id=args.eval_wandb_runid,
                entity=args.entity,
                project=project,
                dir=wandb_dir,
                notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
                # For convenience, keep any 'generate_answers' configs from old run,
                # but overwrite the rest!
                # NOTE: This means any special configs affecting this script must be
                # called again when calling this script!
                config={**old_run.config, **args.__dict__},
                tags=["eval_only", args.experiment_lot, f"entailment={args.entailment_model}",
                      f"metric={args.metric}-{args.metric_model}"],
                resume="allow"
            )
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)
        run.tags = run.tags + (old_exp["args"].model_name, old_exp["args"].dataset, old_exp["args"].brief_prompt)

    else:
        logging.info('Reuse active wandb id.')
        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored
    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore('train_generations.pkl')
            with open(train_generations_pickle.name, 'rb') as infile:
                train_generations = pickle.load(infile)

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    if args.compute_p_true_in_compute_stage:
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)

        logging.info(f"Using {args.metric_model} for computing accuracies")
        if args.metric_model is not None:
            metric_model = utils.init_model_from_name(args.metric_model)
        else:
            metric_model = None

    # Restore outputs from `generate_answers.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []
    result_dict["graphs"] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                try:
                    acc = metric(most_likely_answer['response'], example, metric_model)
                except Exception as e:
                    logging.error("Unable to calculate metric due to an error from the model. Rollback to previous acc")
                    logging.error(str(e))
                    acc = most_likely_answer['accuracy']
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_generations[tid]['most_likely_answer']["accuracy"] = acc
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')
        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)
            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))
            
            # Compute semantic entropy.
            unique_ids, log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            graph = get_entailment_graph(
                responses, model=entailment_model,
                example=example, is_weighted=False
            )
            for k, value in all_graph_entropies(graph):
                entropies[k].append(value)

            weighted_graph = get_entailment_graph(
                responses, model=entailment_model,
                example=example, is_weighted=True
            )
            for k, value in all_graph_entropies(weighted_graph):
                entropies[f"weighted_{k}"].append(value)

            weighted_graph_deberta = get_entailment_graph(
                responses, model=entailment_model,
                example=example, is_weighted=True, weight_strategy="deberta"
            )
            for k, value in all_graph_entropies(weighted_graph):
                entropies[f"weighted_deberta_{k}"].append(value)

            semantic_graph = get_semantic_ids_graph(
                responses, semantic_ids=semantic_ids, ordered_ids=unique_ids, model=entailment_model,
                example=example
            )
            for k, value in all_semantic_entropies(semantic_graph, log_likelihood_per_semantic_id):
                entropies[k].append(value)

            for k, value in all_semantic_entropies_diag(semantic_graph, log_likelihood_per_semantic_id):
                entropies[k].append(value)
            
            for k, value in full_sem_unc_plus_klu(weighted_graph, log_likelihood_per_semantic_id, semantic_ids=semantic_ids, ordered_sem_ids=unique_ids):
                entropies[k].append(value)

            for k, value in full_sem_unc_plus_klu(weighted_graph_deberta, log_likelihood_per_semantic_id, semantic_ids=semantic_ids, ordered_sem_ids=unique_ids):
                entropies[f"deberta_{k}"].append(value)
                
            result_dict["graphs"].append({
                "graph": graph,
                "semantic_graph": semantic_graph,
                "weighted_graph": weighted_graph,
            })

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            #logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break
    if args.recompute_accuracy:
        logging.info('Saving new generations')
        utils.save(validation_generations, f'validation_generations.pkl')
    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict or args.overwrite_metrics:
        logging.info("Initialize uncertainty measures")
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if args.compute_p_ik:
        try:
            logging.info('Starting training p_ik on train embeddings.')
            # Train classifier of correct/incorrect from embeddings.
            p_ik_predictions = get_p_ik(
                train_embeddings=train_embeddings, is_false=train_is_false,
                eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
            result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
            logging.info('Finished training p_ik on train embeddings.')
        except Exception as e:
            logging.error("Cannot comput_p_ik_answerable")
            print(str(e))

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        try:
            p_ik_predictions = get_p_ik(
                train_embeddings=train_embeddings, is_false=train_unanswerable,
                eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        except Exception as e:
            logging.error("Cannot compute_p_ik_answerable")
            print(str(e))
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id, num_test_samples=args.num_test_samples)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
