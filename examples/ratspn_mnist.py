import json
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import deeprob.spn.models as spn
from deeprob.torch.transforms import Flatten
from deeprob.torch.routines import train_model, test_model

if __name__ == '__main__':
    n_features, n_classes = 784, 10

    # Set the preprocessing transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Flatten()
    ])

    # Setup the datasets
    data_train = datasets.MNIST('datasets', train=True, transform=transform, download=True)
    data_test = datasets.MNIST('datasets', train=False, transform=transform, download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = data.random_split(data_train, [n_train, n_val])

    # Instantiate a RAT-SPN model with Gaussian leaves
    ratspn = spn.GaussianRatSpn(
        n_features,
        out_classes=n_classes,  # The number of classes
        rg_depth=3,             # The region graph's depth
        rg_repetitions=8,       # The region graph's number of repetitions
        rg_batch=16,            # The region graph's number of batched leaves
        rg_sum=16,              # The region graph's number of sum nodes per region
        in_dropout=0.2,         # The probabilistic dropout rate to use at leaves layer
        sum_dropout=0.2         # The probabilistic dropout rate to use at sum nodes
    )

    # Train the model using discriminative setting, i.e. by minimizing the categorical cross-entropy
    train_model(
        ratspn, data_train, data_val, setting='discriminative',
        lr=1e-2, batch_size=100, epochs=20, patience=5, checkpoint='checkpoint-ratspn-mnist.pt'
    )

    # Test the model, plotting the test negative log-likelihood and some classification metrics
    nll, metrics = test_model(ratspn, data_test, setting='discriminative')
    print('Test NLL: {:.4f}'.format(nll))
    metrics = json.loads(json.dumps(metrics), parse_float=lambda x: round(float(x), 2))
    print('Test Metrics: {}'.format(json.dumps(metrics, indent=4)))

    # Save the RAT-SPN to file, as any Torch model
    model_filename = 'ratspn-mnist.pt'
    print("Saving model's definition and parameters to {}".format(model_filename))
    torch.save(ratspn.state_dict(), model_filename)
