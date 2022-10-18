import time
import json
import random
import argparse
import numpy as np

from typing import Union, List

from benchmark.utils import deeprob_learn_binary_spn, deeprob_learn_continuous_spn, deeprob_learn_binary_clt

import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
from deeprob.context import ContextState
from deeprob.utils import DataStandardizer

from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS, load_binary_dataset, load_continuous_dataset


def benchmark_log_likelihood(model: Union[spn.Node, spn.BinaryCLT], data: np.ndarray) -> List[float]:
    dts = list()
    if isinstance(model, spn.BinaryCLT):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            model.log_likelihood(data)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
    elif isinstance(model, spn.Node):
        with ContextState(check_spn=False):
            for i in range(args.num_reps):
                start_time = time.perf_counter()
                spnalg.log_likelihood(model, data, n_jobs=args.n_jobs)
                end_time = time.perf_counter()
                dts.append(end_time - start_time)
    else:
        raise ValueError("Unknown model")
    return dts


def benchmark_mpe(model: Union[spn.Node, spn.BinaryCLT], data: np.ndarray) -> List[float]:
    dts = list()
    if isinstance(model, spn.BinaryCLT):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            model.mpe(data)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
    elif isinstance(model, spn.Node):
        with ContextState(check_spn=False):
            for i in range(args.num_reps):
                start_time = time.perf_counter()
                spnalg.mpe(model, data, inplace=False, n_jobs=args.n_jobs)
                end_time = time.perf_counter()
                dts.append(end_time - start_time)
    else:
        raise ValueError("Unknown model")
    return dts


def benchmark_csampling(model: Union[spn.Node, spn.BinaryCLT], data: np.ndarray) -> List[float]:
    dts = list()
    if isinstance(model, spn.BinaryCLT):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            model.sample(data)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
    elif isinstance(model, spn.Node):
        with ContextState(check_spn=False):
            for i in range(args.num_reps):
                start_time = time.perf_counter()
                spnalg.sample(model, data, inplace=False)
                end_time = time.perf_counter()
                dts.append(end_time - start_time)
    else:
        raise ValueError("Unknown model")
    return dts


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="DeeProb-kit Benchmark"
    )
    parser.add_argument(
        'model', choices=['spn', 'binary-clt'], help="The model to benchmark"
    )
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS, help="The dataset"
    )
    parser.add_argument(
        '--num-reps', type=int,  default=1, help="Number of repetitions"
    )
    parser.add_argument(
        '--num-samples', type=int,  default=1000, help="The number of samples (used to benchmark sampling)"
    )
    parser.add_argument(
        '--mar-prob', type=float, default=0.5,
        help="Marginalization probability (used to benchmark marginal queries and sampling)"
    )
    parser.add_argument(
        '--algs', type=str, help="The algorithms to benchmark, separated by a dot:\n"
        + "Complete Evidence (evi), Marginal (mar), Most Probable Explaination (mpe), Conditional Sampling (csampling)",
        default="evi.mar.mpe.csampling"
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1, help="The number of parallel jobs"
    )
    parser.add_argument(
        '--out-filepath', type=str, help="JSON results filepath, defaults to deeprob-{model}-{dataset}.json",
        default=""
    )
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true', help="Whether to enable verbose mode."
    )
    args = parser.parse_args()

    # Check arguments
    if args.model == 'binary-clt' and args.dataset in CONTINUOUS_DATASETS:
        raise ValueError("Cannot benchmark BinaryCLT on a continuous dataset")
    if args.mar_prob <= 0.0 or args.mar_prob >= 1.0:
        raise ValueError("Invalid marginalization probability")

    # Set always the same seed
    random.seed(42)
    np.random.seed(42)

    # Load the dataset
    if args.verbose:
        print(f"Preparing {args.dataset} ...")
    if args.dataset in BINARY_DATASETS:
        data, _, _ = load_binary_dataset(
            '../experiments/datasets', args.dataset, raw=True
        )
    else:
        transform = DataStandardizer()
        data, _, _ = load_continuous_dataset(
            '../experiments/datasets', args.dataset, raw=True, random_state=args.seed
        )
        transform.fit(data)
        data = transform.forward(data)

    # Marginalize some variables randomly with 0.5 probability
    random_state = np.random.RandomState(1234)
    mar_data = data.copy().astype(np.float32, copy=False)
    mar_data[random_state.rand(*mar_data.shape) <= args.mar_prob] = np.nan

    # Initialize the model
    if args.verbose:
        print(f"Initializing {args.model} ...")
    if args.model == 'spn':
        if args.dataset in BINARY_DATASETS:
            model = deeprob_learn_binary_spn(data)
        else:
            model = deeprob_learn_continuous_spn(data)
    elif args.model == 'binary-clt':
        model = deeprob_learn_binary_clt(data)
    else:
        raise ValueError("Unknown model name")

    # The results dictionary
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'num-reps': args.num_reps,
        'num-samples': args.num_samples,
        'n-jobs': args.n_jobs
    }

    # Benchmark algorithms
    for alg in args.algs.split('.'):
        if args.verbose:
            print("Benchmarking {} ...".format(alg))
        if alg == 'evi':
            dts = benchmark_log_likelihood(model, data)
        elif alg == 'mar':
            dts = benchmark_log_likelihood(model, mar_data)
        elif alg == 'mpe':
            dts = benchmark_mpe(model, mar_data)
        elif alg == 'csampling':
            dts = benchmark_csampling(model, mar_data)
        else:
            raise ValueError("Unknown algorithm identifier")
        results[alg] = {'mu_dt': np.mean(dts), 'std_dt': 2.0 * np.std(dts)}

    # Save the benchmark results to file
    out_filepath = args.out_filepath
    if out_filepath == "":
        out_filepath = f"deeprob-{args.model}-{args.dataset}-J{args.n_jobs}.json"
    with open(out_filepath, 'w') as f:
        results = json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 2))
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {out_filepath}")
