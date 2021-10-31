import json
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import deeprob.spn.models as spn
from deeprob.torch.routines import train_model, test_model

if __name__ == '__main__':
    n_features, n_classes = (1, 28, 28), 10

    # Set the preprocessing transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Setup the datasets
    data_train = datasets.MNIST('datasets', train=True, transform=transform, download=True)
    data_test = datasets.MNIST('datasets', train=False, transform=transform, download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = data.random_split(data_train, [n_train, n_val])

    # Instantiate a DGC-SPN model with Gaussian leaves
    dgcspn = spn.DgcSpn(
        n_features,
        out_classes=n_classes,   # The number of classes
        n_batch=16,              # The number of batched leaves
        sum_channels=32,         # The sum layers number of channels
        depthwise=True,          # Use depthwise convolutions at every product layer
        n_pooling=2,             # Then number of initial pooling product layers
        in_dropout=0.2,          # The probabilistic dropout rate to use at leaves layer
        sum_dropout=0.2,         # The probabilistic dropout rate to use at sum layers
        uniform_loc=(-1.5, 1.5)  # Initialize Gaussian locations uniformly
    )

    # Train the model using discriminative setting, i.e. by minimizing the categorical cross-entropy
    train_model(
        dgcspn, data_train, data_val, setting='discriminative',
        lr=1e-2, batch_size=64, epochs=20, patience=3, checkpoint='checkpoint-dgcspn-mnist.pt'
    )

    # Test the model, plotting the test negative log-likelihood and some classification metrics
    nll, metrics = test_model(dgcspn, data_test, batch_size=64, setting='discriminative')
    print('Test NLL: {:.4f}'.format(nll))
    metrics = json.loads(json.dumps(metrics), parse_float=lambda x: round(float(x), 2))
    print('Test Metrics: {}'.format(json.dumps(metrics, indent=4)))

    # Save the model to file
    model_filename = 'dgcspn-mnist.pt'
    print("Saving model's definition and parameters to {}".format(model_filename))
    torch.save(dgcspn.state_dict(), model_filename)
