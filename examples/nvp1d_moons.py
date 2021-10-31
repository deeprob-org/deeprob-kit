import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import deeprob.flows.models as flows
from deeprob.torch.routines import train_model

if __name__ == '__main__':
    # Generate, preprocess and split the moons dataset
    data, _ = make_moons(n_samples=10000, shuffle=True, noise=0.05)
    data = data - [0.5, 0.25]
    data = data.astype(np.float32)
    data_train, data_val = train_test_split(data, test_size=0.2)

    # Instantiate a 1D RealNVP normalizing flow model
    realnvp = flows.RealNVP1d(
        in_features=2,
        n_flows=10,       # The number of transformations
        depth=2,          # The depth of each transformation's conditioner network
        units=128,        # The number of units of conditioner networks hidden layers
        batch_norm=False  # Disable batch normalization, this is a simple task
    )

    # Train the model in the generative setting, i.e. my maximizing the log-likelihood
    train_model(
        realnvp, data_train, data_val, setting='generative',
        lr=1e-4, batch_size=100, epochs=100, patience=5, checkpoint='checkpoint-realnvp-1d-moons.pt'
    )

    # Sample some data points and plot them
    realnvp.eval()  # Make sure to switch to evaluation mode
    samples = realnvp.sample(1000).cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], marker='o', s=2)

    scatter_filename = 'realnvp-moons-scatter.png'
    print("Plotting scatter plot to {} ...".format(scatter_filename))
    plt.savefig(scatter_filename, dpi=192)
