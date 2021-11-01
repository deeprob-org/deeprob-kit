import os
import time
import json
import argparse

from deeprob.utils.data import DataStandardizer
from deeprob.spn.utils.statistics import compute_statistics
from deeprob.spn.structure.leaf import Bernoulli, Gaussian
from deeprob.spn.learning.wrappers import learn_estimator

from experiments.datasets import load_binary_dataset, load_continuous_dataset
from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS
from experiments.utils import evaluate_log_likelihoods


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Vanilla Sum-Product Networks (SPNs) experiments"
    )
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS, help="The dataset."
    )
    parser.add_argument(
        '--learn-leaf', choices=['mle', 'isotonic', 'binary-clt'], default='mle',
        help="The method for leaf learning."
    )
    parser.add_argument(
        '--split-rows', choices=['kmeans', 'kmeans_mb', 'gmm', 'dbscan', 'wald', 'rdc', 'random'], default='gmm',
        help="The splitting rows method."
    )
    parser.add_argument(
        '--split-cols', choices=['gvs', 'rgvs', 'wrgvs', 'ebvs', 'ebvs_ae', 'gbvs', 'gbvs_ag', 'rdc', 'random'],
        default='gvs', help="The splitting columns method."
    )
    parser.add_argument(
        '--min-rows-slice', type=int, default=256, help="The minimum number of rows for splitting."
    )
    parser.add_argument(
        '--min-cols-slice', type=int, default=2, help="The minimum number of columns for splitting."
    )
    parser.add_argument(
        '--n-clusters', type=int, default=2, help="The number of clusters for rows splitting."
    )
    parser.add_argument(
        '--gtest-threshold', type=float, default=5.0, help="The threshold for the G-Test independence test."
    )
    parser.add_argument(
        '--rdc-threshold', type=float, default=0.3, help="The threshold for the RDC independence test."
    )
    parser.add_argument(
        '--ebvs-threshold', type=float, default=0.3, help='The threshold for the Entropy/Gini column splitting'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1, help="The Laplace smoothing value."
    )
    parser.add_argument(
        '--seed', type=int, default=42, help="The seed value to use."
    )
    parser.add_argument(
        '--no-verbose', dest='verbose', action='store_false', help="Whether to disable verbose mode."
    )
    args = parser.parse_args()

    # Load the dataset
    if args.dataset in BINARY_DATASETS:
        data_train, data_valid, data_test = load_binary_dataset(
            'datasets', args.dataset, raw=True
        )
    else:
        transform = DataStandardizer()
        data_train, data_valid, data_test = load_continuous_dataset(
            'datasets', args.dataset, raw=True, random_state=args.seed
        )
        transform.fit(data_train)
        data_train = transform.forward(data_train)
        data_valid = transform.forward(data_valid)
        data_test = transform.forward(data_test)
    _, n_features = data_train.shape

    # Set the data distributions and domains at leaves
    if args.dataset in BINARY_DATASETS:
        distributions = [Bernoulli] * n_features
        domains = [[0, 1]] * n_features
    else:
        distributions = [Gaussian] * n_features
        domains = None  # Automatically detect domains for continuous data

    # Create the results directory
    identifier = time.strftime("%Y%m%d-%H%M%S")
    directory = os.path.join('spn', args.dataset, identifier)
    os.makedirs(directory, exist_ok=True)
    results_filepath = os.path.join(directory, 'results.json')

    # Set the learn leaf method parameters
    learn_leaf_kwargs = dict()
    if args.learn_leaf in ['mle', 'isotonic', 'cltree']:
        learn_leaf_kwargs['alpha'] = args.smoothing

    # Set the split rows method parameters
    split_rows_kwargs = dict()
    if args.split_rows in ['kmeans', 'gmm', 'wald', 'kmeans_mb']:
        split_rows_kwargs['n'] = args.n_clusters

    # Set the split columns method parameters
    split_cols_kwargs = dict()
    if args.split_cols in ['gvs', 'rgvs', 'wrgvs']:
        split_cols_kwargs['p'] = args.gtest_threshold
    elif args.split_cols == 'rdc':
        split_cols_kwargs['d'] = args.rdc_threshold
    elif args.split_cols in ['ebvs', 'gbvs']:
        split_cols_kwargs['alpha'] = args.smoothing
        split_cols_kwargs['e'] = args.ebvs_threshold
    elif args.split_cols in ['ebvs_ae', 'gbvs_ag']:
        split_cols_kwargs['alpha'] = args.smoothing
        split_cols_kwargs['e'] = args.ebvs_threshold
        split_cols_kwargs['size'] = len(data_train)

    # Learn a SPN density estimator
    start_time = time.perf_counter()
    spn = learn_estimator(
        data=data_train,
        distributions=distributions,
        domains=domains,
        learn_leaf=args.learn_leaf,
        split_rows=args.split_rows,
        split_cols=args.split_cols,
        min_rows_slice=args.min_rows_slice,
        min_cols_slice=args.min_cols_slice,
        learn_leaf_kwargs=learn_leaf_kwargs,
        split_rows_kwargs=split_rows_kwargs,
        split_cols_kwargs=split_cols_kwargs,
        random_state=args.seed,
        verbose=args.verbose
    )
    learning_time = time.perf_counter() - start_time

    # Compute the log-likelihoods for the validation and test datasets
    valid_mean_ll, valid_stddev_ll = evaluate_log_likelihoods(spn, data_valid)
    test_mean_ll, test_stddev_ll = evaluate_log_likelihoods(spn, data_test)

    # Save the results
    results = {
        'log_likelihood': {
            'valid': {'mean': valid_mean_ll, 'stddev': valid_stddev_ll},
            'test': {'mean': test_mean_ll, 'stddev': test_stddev_ll}
        },
        'learning_time': learning_time,
        'statistics': compute_statistics(spn),
        'settings': args.__dict__
    }
    with open(results_filepath, 'w') as f:
        json.dump(results, f, indent=4)
