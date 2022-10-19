import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import deeprob.spn.models as spn
import deeprob.flows.models as flows
from deeprob.torch.transforms import TransformList, Flatten
from deeprob.torch.datasets import WrappedDataset
from deeprob.torch.routines import train_model, test_model

if __name__ == '__main__':
    # Load the MNIST dataset and split the dataset
    in_shape = (1, 28, 28)
    in_features = np.prod(in_shape).item()
    data_train = datasets.MNIST('datasets', train=True, transform=transforms.ToTensor(), download=True)
    data_test = datasets.MNIST('datasets', train=False, transform=transforms.ToTensor(), download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = data.random_split(data_train, [n_train, n_val])

    # Set the preprocessing transformation, i.e. flatten
    transform = TransformList([
        Flatten(in_shape)  # Specify the input shape to compute the inverse of the transformation
    ])

    # Wrap MNIST datasets for unsupervised setting
    data_train = WrappedDataset(data_train, unsupervised=True, transform=transform)
    data_val = WrappedDataset(data_val, unsupervised=True, transform=Flatten())
    data_test = WrappedDataset(data_test, unsupervised=True, transform=Flatten())

    # Instantiate a RAT-SPN model with Gaussian leaves
    ratspn = spn.GaussianRatSpn(
        in_features,
        rg_depth=3,        # The region graph's depth
        rg_repetitions=8,  # The region graph's number of repetitions
        rg_batch=16,       # The region graph's number of batched leaves
        rg_sum=16          # The region graph's number of sum nodes per region
    )

    # Instantiate a RealNVP 1d model with the RAT-SPN model as base distribution
    realnvp = flows.RealNVP1d(
        in_features,
        dequantize=True,
        logit=1e-6,
        in_base=ratspn,
        n_flows=5,
        depth=2,
        units=512,
        batch_norm=True,
        affine=True
    )

    # Train the model using generative setting, i.e. by maximizing the log-likelihood
    train_model(
        realnvp, data_train, data_val, setting='generative',
        lr=1e-3, batch_size=100, epochs=10, patience=3, checkpoint='checkpoint-ratspn-realnvp-mnist.pt'
    )

    # Test the model using generative setting
    # Also, compute bits-per-dimension
    mu_ll, sigma_ll = test_model(realnvp, data_test, setting='generative')
    bpp = (-mu_ll / np.log(2)) / in_features
    print('Mean LL: {:.4f} - Two Stddev LL: {:.4f} - Bits per Dimension: {:.2f}'.format(mu_ll, sigma_ll, bpp))

    # Sample some data points and plot them
    realnvp.eval()  # Make sure to switch to evaluation mode
    n_samples = 10
    samples = realnvp.sample(n_samples ** 2).cpu()
    images = torch.stack([transform.backward(x) for x in samples])
    samples_filename = 'ratspn-realnvp-mnist-samples.png'
    print("Plotting generated samples to {} ...".format(samples_filename))
    utils.save_image(images, samples_filename, nrow=n_samples, padding=0)

    # Save the model to file, as any Torch model
    model_filename = 'ratspn-realnvp-mnist.pt'
    print("Saving model's definition and parameters to {}".format(model_filename))
    torch.save(realnvp.state_dict(), model_filename)
