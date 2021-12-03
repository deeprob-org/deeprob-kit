import os
import time
import json
import argparse
import torch

from deeprob.flows.models.maf import MAF
from deeprob.flows.models.realnvp import RealNVP1d, RealNVP2d
from deeprob.utils.statistics import compute_bpp
from deeprob.torch.metrics import fid_score

from experiments.datasets import load_continuous_dataset, load_vision_dataset
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_samples, save_grid_images


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Density Estimation with Normalizing Flows experiments'
    )
    parser.add_argument(
        'dataset', choices=CONTINUOUS_DATASETS + VISION_DATASETS, help="The dataset."
    )
    parser.add_argument(
        '--model', choices=['realnvp-1d', 'realnvp-2d', 'maf'], default='realnvp-1d',
        help="The normalizing flow model to use."
    )
    parser.add_argument(
        '--random-hflip', action='store_true', help="Whether to apply random horizontal flip for vision datasets."
    )
    parser.add_argument(
        '--logit', type=float, default=None,
        help="The logit value to use for vision datasets. If not None also dequantization will be used."
    )
    parser.add_argument(
        '--n-flows', type=int, default=5,
        help="The number of normalizing flows layers. \
            For RealNVP-2D this corresponds to the number of multi-scale architecture repetitions."
    )
    parser.add_argument(
        '--no-batch-norm', dest='batch_norm', action='store_false',
        help="Whether to disable batch normalization for RealNVP-1D and MAF."
    )
    parser.add_argument(
        '--network', choices=['resnet', 'densenet'], default='resnet',
        help="The RealNVP-2D conditioner neural network architecture"
    )
    parser.add_argument(
        '--depth', type=int, default=1,
        help="The depth of each dense conditioner network."
    )
    parser.add_argument(
        '--units', type=int, default=128,
        help="The number of units of each dense conditioner network."
    )
    parser.add_argument(
        '--channels', type=int, default=32,
        help="The number of channels in convolutional conditioner networks."
    )
    parser.add_argument(
        '--n-blocks', type=int, default=2,
        help='The number of blocks in convolutional conditioner networks.'
    )
    parser.add_argument(
        '--no-affine', dest='affine', action='store_false',
        help="Whether to use only translations instead of affine transformations (as in NICE)."
    )
    parser.add_argument(
        '--activation', choices=['relu', 'tanh', 'sigmoid'], default='relu',
        help="The activation function to use in MAF conditioner networks."
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
    is_vision_dataset = args.dataset in VISION_DATASETS
    if is_vision_dataset:
        data_train, data_valid, data_test = load_vision_dataset(
            'datasets', args.dataset,
            unsupervised=True,
            standardize=not args.logit,
            flatten=args.model in ['realnvp-1d', 'maf'],
            random_hflip=args.random_hflip,
            random_state=args.seed
        )
    else:
        data_train, data_valid, data_test = load_continuous_dataset(
            'datasets', args.dataset, random_state=args.seed
        )
    in_features = data_train.features_shape

    # Create the results directory
    identifier = time.strftime("%Y%m%d-%H%M")
    directory = os.path.join(args.model, args.dataset, identifier)
    os.makedirs(directory, exist_ok=True)
    results_filepath = os.path.join(directory, 'results.json')
    samples_filepath = os.path.join(directory, 'samples.png')
    checkpoint_filepath = os.path.join(directory, 'checkpoint.pt')

    if args.model == 'realnvp-1d':
        model = RealNVP1d(
            in_features,
            dequantize=args.logit is not None,
            logit=args.logit,
            n_flows=args.n_flows,
            depth=args.depth,
            units=args.units,
            batch_norm=args.batch_norm,
            affine=args.affine
        )
    elif args.model == 'realnvp-2d':
        model = RealNVP2d(
            in_features,
            dequantize=args.logit is not None,
            logit=args.logit,
            network=args.network,
            n_flows=args.n_flows,
            n_blocks=args.n_blocks,
            channels=args.channels,
            affine=args.affine
        )
    elif args.model == 'maf':
        model = MAF(
            in_features,
            dequantize=args.logit is not None,
            logit=args.logit,
            n_flows=args.n_flows,
            depth=args.depth,
            units=args.units,
            batch_norm=args.batch_norm,
            activation=args.activation,
            sequential=in_features <= args.units
        )
    else:
        raise NotImplementedError("Experiments for model {} are not implemented".format(args.model))

    # Train the model and collect the results
    mean_ll, stddev_ll = collect_results_generative(
        model, data_train, data_valid, data_test,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        optimizer='adam',
        optimizer_kwargs={'weight_decay': args.weight_decay},
        checkpoint=checkpoint_filepath,
        verbose=args.verbose
    )

    # Sample some images, if necessary
    if is_vision_dataset:
        images = collect_samples(model, 100)
        if data_train.transform is not None:
            images = torch.stack([data_train.transform.backward(x) for x in images])
        save_grid_images(images, samples_filepath)

    # Compute BPP and FID scores, if necessary
    bpp, fid = None, None
    if is_vision_dataset:
        bpp = compute_bpp(mean_ll, data_train.features_shape)
        if args.model in ['realnvp-2d']:
            samples = collect_samples(model, 5000, batch_size=args.batch_size)
            del model  # Delete the model to reserve some extra memory required to compute the FID score
            fid = fid_score(data_test, samples, batch_size=max(1, args.batch_size // 4))

    # Save the results
    results = {
        'log_likelihood': {'mean': mean_ll, 'stddev': stddev_ll},
        'bpp': bpp, 'fid': fid, 'settings': args.__dict__
    }
    with open(results_filepath, 'w') as f:
        json.dump(results, f, indent=4)
