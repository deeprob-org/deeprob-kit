import gc
import time
import numpy as np

from typing import Union, List
from utils import compute_mean_stddev_times, load_dataset

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Node, Sum, Product, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Bernoulli

from deeprob.context import ContextState
import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
import deeprob.spn.utils as spnutils


def learn_binary_clt(data: np.ndarray) -> spn.BinaryCLT:
    n_features = data.shape[1]
    scope = list(range(n_features))
    domain = [[0, 1]] * n_features
    clt = spn.BinaryCLT(scope, root=0)
    clt.fit(data, domain, alpha=0.1, random_state=42)
    return clt


def deeprob_build_spn(clt: spn.BinaryCLT) -> spn.Node:
    return clt.to_pc()


def spflow_build_spn(clt: spn.BinaryCLT) -> Node:  # This function is copy-pasted from BinaryCLT.to_pc()
    from deeprob.utils.graph import build_tree_structure
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
    n_repetitions = 20
    data_train, evi_data, mar_data = load_dataset('ad', n_samples=5000, seed=42)
    clt = learn_binary_clt(data_train)

    root = deeprob_build_spn(clt)
    print(spnutils.compute_statistics(root))

    # ---- DeeProb-kit EVI Inference ----------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    with ContextState(check_spn=False):
        for _ in range(n_repetitions):
            start_time = time.perf_counter()
            lls = spnalg.log_likelihood(root, evi_data)
            end_time = time.perf_counter()
            elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[EVI] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- DeeProb-kit MAR Inference ----------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    with ContextState(check_spn=False):
        for _ in range(n_repetitions):
            start_time = time.perf_counter()
            lls = spnalg.log_likelihood(root, mar_data)
            end_time = time.perf_counter()
            elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MAR] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- DeeProb-kit MPE Inference ----------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    with ContextState(check_spn=False):
        for _ in range(n_repetitions):
            start_time = time.perf_counter()
            mpe_data = spnalg.mpe(root, mar_data, inplace=False)
            end_time = time.perf_counter()
            elapsed_times.append(1e3 * (end_time - start_time))
    lls = spnalg.log_likelihood(root, mpe_data)
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MPE] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- DeeProb-kit Sampling ---------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    with ContextState(check_spn=False):
        for _ in range(n_repetitions):
            start_time = time.perf_counter()
            _ = spnalg.sample(root, mar_data, inplace=False)
            end_time = time.perf_counter()
            elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[SMP] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f})".format(
        mean_time, stddev_time
    ))

    print()

    root = spflow_build_spn(clt)

    # ---- SPFlow EVI Inference ---------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        lls = log_likelihood(root, evi_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[EVI] SPFlow Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- SPFlow MAR Inference ---------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        lls = log_likelihood(root, mar_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MAR] SPFlow Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- SPFlow MPE Inference ---------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        mpe_data = mpe(root, mar_data, in_place=False)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    lls = log_likelihood(root, mpe_data, dtype=np.float32)
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MPE] SPFlow Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- SPFlow Sampling --------------------------------------------------------------------------------------------
    gc.collect()
    rand_gen = np.random.RandomState(42)
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        _ = sample_instances(root, mar_data, rand_gen=rand_gen, in_place=False)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[SMP] SPFlow Avg. Time: {:.2f}ms (+- {:.2f})".format(
        mean_time, stddev_time
    ))

    print()
