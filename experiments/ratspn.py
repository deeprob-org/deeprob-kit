import os
import time
import json
import argparse
import torch

from deeprob.spn.models.ratspn import GaussianRatSpn, BernoulliRatSpn
from deeprob.utils.statistics import compute_bpp

from experiments.datasets import load_binary_dataset, load_continuous_dataset, load_vision_dataset
from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_samples
from experiments.utils import save_grid_images


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Randomized And Tensorized Sum-Product Networks (RAT-SPNs) experiments"
    )
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS + VISION_DATASETS, help="The dataset."
    )
    parser.add_argument(
        '--discriminative', action='store_true', help="Whether to use discriminative settings (for vision datasets)."
    )
    parser.add_argument(
        '--random-hflip', action='store_true', help="Whether to apply random horizontal flip for vision datasets."
    )
    parser.add_argument(
        '--rg-depth', type=int, default=1, help="The region graph's depth."
    )
    parser.add_argument(
        '--rg-repetitions', type=int, default=4, help="The region graph number of repetitions."
    )
    parser.add_argument(
        '--rg-batch', type=int, default=8, help="The region graph's number of distribution batches."
    )
    parser.add_argument(
        '--rg-sum', type=int, default=8, help="The region graph's' number of sum nodes per region."
    )
    parser.add_argument(
        '--uniform-loc', type=float, nargs=2, default=None,
        help="Sample locations uniformly for input distributions layer initialization."
    )
    parser.add_argument(
        '--in-dropout', type=float, default=None, help="The input distributions layer dropout to use."
    )
    parser.add_argument(
        '--sum-dropout', type=float, default=None, help="The probabilistic dropout to use."
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3, help="The learning rate."
    )
    parser.add_argument(
        '--batch-size', type=int, default=100, help="The batch size."
    )
    parser.add_argument(
        '--epochs', type=int, default=1000, help="The number of epochs."
    )
    parser.add_argument(
        '--patience', type=int, default=30, help="The epochs patience used for early stopping."
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0, help="The L2 regularization factor."
    )
    parser.add_argument(
        '--seed', type=int, default=42, help="The seed value to use."
    )
    parser.add_argument(
        '--no-verbose', dest='verbose', action='store_false', help="Whether to disable verbose mode."
    )
    args = parser.parse_args()

    # Check dataset and --discriminative flag
    is_vision_dataset = args.dataset in VISION_DATASETS
    is_binary_dataset = args.dataset in BINARY_DATASETS
    is_continuous_dataset = args.dataset in CONTINUOUS_DATASETS
    assert is_vision_dataset or args.discriminative is False, \
        "Discriminative setting is not supported for dataset {}".format(args.dataset)

    # Load the dataset
    if is_binary_dataset:
        data_train, data_valid, data_test = load_binary_dataset(
            'datasets', args.dataset
        )
    elif is_continuous_dataset:
        data_train, data_valid, data_test = load_continuous_dataset(
            'datasets', args.dataset, random_state=args.seed
        )
    else:
        data_train, data_valid, data_test = load_vision_dataset(
            'datasets', args.dataset,
            unsupervised=not args.discriminative,
            flatten=True,
            standardize=True,
            random_hflip=args.random_hflip,
            random_state=args.seed
        )
    in_features = data_train.features_shape
    out_classes = data_train.num_classes if args.discriminative else 1

    # Create the results directory
    identifier = time.strftime("%Y%m%d-%H%M")
    if args.discriminative:
        directory = os.path.join('ratspn', 'discriminative', args.dataset, identifier)
        os.makedirs(directory, exist_ok=True)
        samples_filepath = None
    else:
        directory = os.path.join('ratspn', 'generative', args.dataset, identifier)
        os.makedirs(directory, exist_ok=True)
        samples_filepath = os.path.join(directory, 'samples.png')
    results_filepath = os.path.join(directory, 'results.json')
    checkpoint_filepath = os.path.join(directory, 'checkpoint.pt')

    # Build the model
    if is_binary_dataset:
        model = BernoulliRatSpn(
            in_features,
            out_classes=out_classes,
            rg_depth=args.rg_depth,
            rg_repetitions=args.rg_repetitions,
            rg_batch=args.rg_batch,
            rg_sum=args.rg_sum,
            in_dropout=args.in_dropout,
            sum_dropout=args.sum_dropout,
            random_state=args.seed
        )
    else:
        model = GaussianRatSpn(
            in_features,
            out_classes=out_classes,
            rg_depth=args.rg_depth,
            rg_repetitions=args.rg_repetitions,
            rg_batch=args.rg_batch,
            rg_sum=args.rg_sum,
            in_dropout=args.in_dropout,
            sum_dropout=args.sum_dropout,
            uniform_loc=args.uniform_loc,
            random_state=args.seed
        )

    if args.discriminative:
        nll, metrics = collect_results_discriminative(
            model, data_train, data_valid, data_test,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            optimizer='adam',
            optimizer_kwargs={'weight_decay': args.weight_decay},
            checkpoint=os.path.join(directory, 'checkpoint.pt'),
            verbose=args.verbose
        )

        # Save the results
        results = {
            'nll': nll, 'metrics': metrics,
            'settings': args.__dict__
        }
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        mean_ll, stddev_ll = collect_results_generative(
            model, data_train, data_valid, data_test,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            optimizer='adam',
            optimizer_kwargs={'weight_decay': args.weight_decay},
            checkpoint=os.path.join(directory, 'checkpoint.pt'),
            verbose=args.verbose
        )

        # Sample and save some images
        if is_vision_dataset:
            images = collect_samples(model, 100)
            if data_train.transform is not None:
                images = torch.stack([data_train.transform.backward(x) for x in images])
            save_grid_images(images, samples_filepath)

        # Compute BPP score, if necessary
        bpp = None
        if is_vision_dataset:
            bpp = compute_bpp(mean_ll, data_train.features_shape)

        # Save the results
        results = {
            'log_likelihood': {'mean': mean_ll, 'stddev': stddev_ll},
            'bpp': bpp, 'settings': args.__dict__
        }
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
