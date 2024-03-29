import gc
import time
import json
import random
import argparse
import numpy as np

from typing import Union, List, Tuple

from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Node
from spn.structure.leaves.cltree.CLTree import CLTree

from benchmark.utils import spflow_learn_binary_spn, spflow_learn_continuous_spn, spflow_learn_binary_clt

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.structure.leaves.cltree.Inference import cltree_log_likelihood
from spn.structure.leaves.cltree.MPE import cltree_mpe

from deeprob.utils import DataStandardizer

from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS, load_binary_dataset, load_continuous_dataset


def benchmark_log_likelihood(model: Union[Node, CLTree], data: np.ndarray) -> Tuple[List[float], np.ndarray]:
    dts = list()
    if isinstance(model, CLTree):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            cltree_log_likelihood(model, data, dtype=np.float32)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
        lls = cltree_log_likelihood(model, data, dtype=np.float32)
    elif isinstance(model, Node):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            log_likelihood(model, data, dtype=np.float32)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
        lls = log_likelihood(model, data, dtype=np.float32)
    else:
        raise ValueError("Unknown model")
    return dts, lls


def benchmark_mpe(model: Union[Node, CLTree], data: np.ndarray) -> Tuple[List[float], np.ndarray]:
    dts = list()
    if isinstance(model, CLTree):
        logprobs = np.empty(len(mar_data), dtype=np.float32)
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            mar_data_copied = mar_data.copy()
            cltree_mpe(model, mar_data_copied, logprobs=logprobs)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
        mpe_data = mar_data.copy()
        cltree_mpe(model, mpe_data, logprobs=logprobs)
        lls = cltree_log_likelihood(model, mpe_data.astype(np.int64), dtype=np.float32)
    elif isinstance(model, Node):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            mpe(model, data, in_place=False)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
        mpe_data = mpe(model, data, in_place=False)
        lls = log_likelihood(model, mpe_data, dtype=np.float32)
    else:
        raise ValueError("Unknown model")
    return dts, lls


def benchmark_csampling(model: Node, data: np.ndarray) -> Tuple[List[float], np.ndarray]:
    assert not isinstance(model, CLTree)
    dts = list()
    rand_gen = np.random.RandomState(42)
    if isinstance(model, Node):
        for i in range(args.num_reps):
            start_time = time.perf_counter()
            sample_instances(model, data, rand_gen=rand_gen, in_place=False)
            end_time = time.perf_counter()
            dts.append(end_time - start_time)
        sampled_data = sample_instances(model, data, rand_gen=rand_gen, in_place=False)
        lls = log_likelihood(model, sampled_data, dtype=np.float32)
    else:
        raise ValueError("Unknown model")
    return dts, lls


def benchmark_learnclt(data: np.ndarray) -> List[float]:
    dts = list()
    for i in range(args.num_reps):
        start_time = time.perf_counter()
        spflow_learn_binary_clt(data)
        end_time = time.perf_counter()
        dts.append(end_time - start_time)
    return dts


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="SPFlow==0.0.41 Benchmark"
    )
    parser.add_argument(
        'model', choices=['spn', 'binary-clt'], help="The model to benchmark"
    )
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS, help="The dataset"
    )
    parser.add_argument(
        '--num-reps', type=int,  default=10, help="Number of repetitions"
    )
    parser.add_argument(
        '--mar-prob', type=float, default=0.5,
        help="Marginalization probability (used to benchmark marginal queries and sampling)"
    )
    parser.add_argument(
        '--algs', type=str, help="The algorithms to benchmark, separated by a dot:\n"
        + "Complete Evidence (evi), Marginal (mar), Most Probable Explaination (mpe), "
        + "Conditional Sampling (csampling), Learn Chow-Liu Tree (learnclt)",
        default="evi.mar.mpe.csampling"
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
        raise ValueError("Cannot benchmark Binary-CLT on a continuous dataset")
    if args.model != 'binary-clt' and 'learnclt' in args.algs:
        raise ValueError("Cannot benchmark `learnclt` algorithm on a non Binary-CLT model")
    if args.model == 'binary-clt' and 'csampling' in args.algs:
        raise ValueError("Cannot benchmark `csampling` algorithm on a Binary-CLT model")
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
            model = spflow_learn_binary_spn(data)
        else:
            model = spflow_learn_continuous_spn(data)
    elif args.model == 'binary-clt':
        model = spflow_learn_binary_clt(data)
    else:
        raise ValueError("Unknown model name")

    # The results dictionary
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'num-reps': args.num_reps
    }

    # Benchmark algorithms
    for alg in args.algs.split('.'):
        if args.verbose:
            print("Benchmarking {} ...".format(alg))
        gc.disable()
        if alg == 'evi':
            dts, lls = benchmark_log_likelihood(model, data)
        elif alg == 'mar':
            dts, lls = benchmark_log_likelihood(model, mar_data)
        elif alg == 'mpe':
            dts, lls = benchmark_mpe(model, mar_data)
        elif alg == 'csampling':
            dts, lls = benchmark_csampling(model, mar_data)
        elif alg == 'learnclt':
            dts = benchmark_learnclt(data)
        else:
            raise ValueError("Unknown algorithm identifier")
        gc.enable()
        results[alg] = {'dt': {'mu': np.mean(dts), 'std': 2.0 * np.std(dts)}}
        if alg not in ['learnclt']:
            results[alg].update({'ll': {'mu': np.mean(lls).item(), 'std': 2.0 * np.std(lls).item()}})

    # Save the benchmark results to file
    out_filepath = args.out_filepath
    if out_filepath == "":
        out_filepath = f"spflow==0.0.41-{args.model}-{args.dataset}.json"
    with open(out_filepath, 'w') as f:
        results = json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 2))
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {out_filepath}")
