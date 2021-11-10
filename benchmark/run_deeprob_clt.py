import gc
import time
import json
import numpy as np

from utils import compute_mean_stddev_times, load_benchmark_data

import deeprob.spn.structure as spn


def learn_binary_clt(data: np.ndarray) -> spn.BinaryCLT:
    n_features = data.shape[1]
    scope = list(range(n_features))
    domain = [[0, 1]] * n_features
    clt = spn.BinaryCLT(scope, root=0)
    clt.fit(data, domain, alpha=0.1, random_state=42)
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

    print("Benchmarking Binary-CLTs on DeeProb-kit ...")
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
        clt.log_likelihood(evi_data)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['evi'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking MAR Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        clt.log_likelihood(mar_data)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['mar'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking MPE Inference ...")
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        clt.mpe(mar_data)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['mpe'] = compute_mean_stddev_times(elapsed_times)

    print("\tBenchmarking Ancestral Sampling ...")
    mis_data = np.full((n_samples, n_features), fill_value=np.nan, dtype=np.float32)
    elapsed_times = list()
    for _ in range(n_repetitions):
        gc.collect()
        start_time = time.perf_counter()
        clt.sample(mar_data)
        end_time = time.perf_counter()
        elapsed_times.append(end_time - start_time)
    results['anc_sampling'] = compute_mean_stddev_times(elapsed_times)

    # Save the benchmark results to file
    results_filepath = 'deeprob-clt-benchmark.json'
    with open(results_filepath, 'w') as f:
        results = json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 2))
        json.dump({'deeprob': {'clt': results}}, f, indent=2)
    print("Saved benchmark results to {}".format(results_filepath))
    print()
