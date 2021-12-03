import os
import time
import json
import argparse
import torch
import numpy as np

from deeprob.spn.models.dgcspn import DgcSpn
from deeprob.utils.statistics import compute_mean_quantiles, compute_bpp

from experiments.datasets import VISION_DATASETS, load_vision_dataset
from experiments.utils import collect_results_generative, collect_results_discriminative
from experiments.utils import collect_image_completions, save_grid_images


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments"
    )
    parser.add_argument(
        'dataset', choices=VISION_DATASETS, help="The vision dataset."
    )
    parser.add_argument(
        '--random-hflip', action='store_true', help="Whether to apply random horizontal flip for vision datasets."
    )
    parser.add_argument(
        '--discriminative', action='store_true', help="Whether to use discriminative settings."
    )
    parser.add_argument(
        '--n-batch', type=int, default=8, help="The number of input distribution layer batches."
    )
    parser.add_argument(
        '--sum-channels', type=int, default=8, help="The number of channels at sum layers."
    )
    parser.add_argument(
        '--depthwise', type=lambda s: False if s == 'False' else True, nargs='+', default=[True],
        help="The flags representing which product layer uses depthwise convolutions."
    )
    parser.add_argument(
        '--n-pooling', type=int, default=0, help="The number of initial pooling product layers."
    )
    parser.add_argument(
        '--quantiles-loc', action='store_true', default=False,
        help="Whether to use mean quantiles for leaves initialization."
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

    # Load the dataset
    data_train, data_valid, data_test = load_vision_dataset(
        'datasets', args.dataset,
        unsupervised=not args.discriminative,
        flatten=False,
        standardize=True,
        random_hflip=args.random_hflip,
        random_state=args.seed
    )
    in_features = data_train.features_shape
    out_classes = data_train.num_classes if args.discriminative else 1

    # Create the results directory
    identifier = time.strftime("%Y%m%d-%H%M")
    if args.discriminative:
        directory = os.path.join('dgcspn', 'discriminative', args.dataset, identifier)
        os.makedirs(directory, exist_ok=True)
        completions_filepath = None
    else:
        directory = os.path.join('dgcspn', 'generative', args.dataset, identifier)
        os.makedirs(directory, exist_ok=True)
        completions_filepath = os.path.join(directory, 'completions.png')
    results_filepath = os.path.join(directory, 'results.json')
    checkpoint_filepath = os.path.join(directory, 'checkpoint.pt')

    # Compute mean quantiles locations, if specified
    quantiles_loc = None
    if args.quantiles_loc:
        if args.discriminative:
            preproc_data = np.asarray([x.numpy() for x, _ in data_train])
        else:
            preproc_data = np.asarray([x.numpy() for x in data_train])
        quantiles_loc = compute_mean_quantiles(preproc_data, args.n_batch)

    # Build the model
    model = DgcSpn(
        in_features,
        out_classes=out_classes,
        n_batch=args.n_batch,
        sum_channels=args.sum_channels,
        depthwise=args.depthwise,
        n_pooling=args.n_pooling,
        in_dropout=args.in_dropout,
        sum_dropout=args.sum_dropout,
        quantiles_loc=quantiles_loc,
        uniform_loc=args.uniform_loc
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

        # Make image completions and save them
        images = collect_image_completions(model, data_test, n_samples=10, random_state=args.seed)
        if data_train.transform is not None:
            images = torch.stack([data_train.transform.backward(x) for x in images])
        save_grid_images(images, completions_filepath, nrow=5)

        # Compute BPP score
        bpp = compute_bpp(mean_ll, data_train.features_shape)

        # Save the results
        results = {
            'log_likelihood': {'mean': mean_ll, 'stddev': stddev_ll},
            'bpp': bpp, 'settings': args.__dict__
        }
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
