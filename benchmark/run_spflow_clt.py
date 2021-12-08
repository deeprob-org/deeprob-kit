import gc
import time
import json
import numpy as np

from utils import compute_mean_stddev_times, load_benchmark_data

from spn.structure.leaves.cltree.CLTree import CLTree
from spn.structure.leaves.cltree.MLE import update_cltree_parameters_mle
from spn.structure.leaves.cltree.Inference import cltree_log_likelihood
from spn.structure.leaves.cltree.MPE import cltree_mpe
from spn.structure.leaves.cltree.Sampling import sample_cltree_node


def learn_binary_clt(data: np.ndarray) -> CLTree:
    n_features = data.shape[1]
    scope = list(range(n_features))
    clt = CLTree(scope, data)
    update_cltree_parameters_mle(clt, data, alpha=0.1)
    return clt


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

    print("Benchmarking Binary-CLTs on SPFlow ...")
    clt = learn_binary_clt(train_data)

    print("\tBenchmarking Chiow-Liu Algorithm ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        learn_binary_clt(train_data)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['chowliu'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking EVI Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        cltree_log_likelihood(clt, evi_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['evi'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking MAR Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        cltree_log_likelihood(clt, mar_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['mar'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking MPE Inference ...")
    logprobs = np.empty(len(mar_data), dtype=np.float32)
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        mar_data_copied = mar_data.copy()
        cltree_mpe(clt, mar_data_copied, logprobs=logprobs)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['mpe'] = compute_mean_stddev_times(elapsed_times)

    # The implementation of ancestral sampling on CLTs in SPFlow have an infinite loop
    results['anc_sampling'] = [None, None]
    """
    print("\tBenchmarking Ancestral Sampling ...")
    mis_data = np.full((n_samples, n_features), fill_value=np.nan, dtype=np.float32)
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        sample_cltree_node(clt, n_samples=n_samples, data=None, rand_gen=rand_gen)  # data is not used ...
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['anc_sampling'] = compute_mean_stddev_times(elapsed_times)
    """

    # Save the benchmark results to file
    results_filepath = 'spflow-clt-benchmark.json'
    with open(results_filepath, 'w') as f:
        results = json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 2))
        json.dump({'spflow': {'clt': results}}, f, indent=2)
    print("Saved benchmark results to {}".format(results_filepath))
    print()
