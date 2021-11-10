import gc
import time
import json
from typing import Union, List
import numpy as np

from utils import compute_mean_stddev_times, load_benchmark_data

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Node, Sum, Product, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Bernoulli

from deeprob.utils.graph import build_tree_structure
import deeprob.spn.structure as spn


def learn_binary_clt(data: np.ndarray) -> spn.BinaryCLT:
    n_features = data.shape[1]
    scope = list(range(n_features))
    domain = [[0, 1]] * n_features
    clt = spn.BinaryCLT(scope, root=0)
    clt.fit(data, domain, alpha=0.1, random_state=42)
    return clt


def learn_binary_spn(data: np.ndarray) -> Node:  # This function is copy-pasted from DeeProb-kit BinaryCLT.to_pc()
    clt = learn_binary_clt(data)
    root = build_tree_structure(clt.tree, scope=clt.scope)
    factors = {clt.scope[i]: np.exp(clt.params[i]) for i in range(len(clt.tree))}
    neg_buffer, pos_buffer = [], []
    nodes_stack = [root]
    last_node_visited = None
    while nodes_stack:
        node = nodes_stack[-1]
        if node.is_leaf() or (last_node_visited in node.get_children()):
            leaves: List[Union[Bernoulli, Sum]] = [
                Bernoulli(p=0.0, scope=node.get_id()),
                Bernoulli(p=1.0, scope=node.get_id()),
            ]
            if not node.is_leaf():
                neg_prod = Product(children=[leaves[0]] + neg_buffer[-len(node.get_children()):])
                pos_prod = Product(children=[leaves[1]] + pos_buffer[-len(node.get_children()):])
                del neg_buffer[-len(node.get_children()):]
                del pos_buffer[-len(node.get_children()):]
                sum_children = [neg_prod, pos_prod]
            else:
                sum_children = leaves
            weights = factors[node.get_id()]
            neg_buffer.append(Sum(children=sum_children, weights=weights[0]))
            pos_buffer.append(Sum(children=sum_children, weights=weights[1]))
            last_node_visited = nodes_stack.pop()
        else:
            nodes_stack.extend(node.get_children())
    return rebuild_scopes_bottom_up(assign_ids(pos_buffer[0]))


if __name__ == '__main__':
    with open('benchmark/settings.json', 'r') as f:
        settings = json.load(f)
    dataset = settings['dataset']
    n_repetitions = settings['n_repetitions']
    n_samples = settings['n_samples']
    results = dict()

    train_data, evi_data, mar_data = load_benchmark_data(dataset, seed=42)
    n_features = train_data.shape[1]

    gc.collect()
    gc.disable()

    print("Benchmarking SPNs on SPFlow ...")
    root = learn_binary_spn(train_data)

    print("\tBenchmarking EVI Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        log_likelihood(root, evi_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['evi'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking MAR Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        log_likelihood(root, mar_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['mar'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking MPE Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        mpe(root, mar_data, in_place=False)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['mpe'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking Conditional Sampling ...")
    rand_gen = np.random.RandomState(42)
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        sample_instances(root, mar_data, rand_gen=rand_gen, in_place=False)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['cond_sampling'] = compute_mean_stddev_times(elapsed_times)

    # Save the benchmark results to file
    results_filepath = 'spflow-spn-benchmark.json'
    with open(results_filepath, 'w') as f:
        results = json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 2))
        json.dump({'spflow': {'spn': results}}, f, indent=2)
    print("Saved benchmark results to {}".format(results_filepath))
    print()
