import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

import deeprob.spn.structure as spn

if __name__ == '__main__':
    # Setup the (binarized) digits dataset
    data, _ = load_digits(return_X_y=True)
    data[data < 8] = 0
    data[data >= 8] = 1
    data = data.astype(np.float32)
    n_samples, n_features = data.shape

    # Initialize the scope and domains
    scope = list(range(n_features))
    domains = [[0, 1]] * n_features

    # Instantiate the random state
    random_state = np.random.RandomState(42)

    # Instantiate and fit a Binary Chow-Liu Tree (CLT)
    clt = spn.BinaryCLT(scope)
    clt.fit(data, domains, alpha=0.01, random_state=random_state)

    # Plot the CLT
    clt_filename = 'clt-bdigits.svg'
    print("Plotting the learnt CLT to {} ...".format(clt_filename))
    spn.plot_binary_clt(clt, clt_filename, show_weights=False)

    # Evaluate the model by computing the average log-likelihood with two standard deviations
    lls = clt.log_likelihood(data)
    mean_ll = np.mean(lls)
    stddev_ll = 2.0 * np.std(lls) / np.sqrt(len(lls))
    print('EVI -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(mean_ll, stddev_ll))

    # Randomly set NaNs to marginalize random variables
    mar_data = data.copy()
    mask = random_state.rand(*mar_data.shape) < 0.2
    mar_data[mask] = np.nan

    # Compute marginalized (MAR) queries
    lls = clt.log_likelihood(mar_data)
    mean_ll = np.mean(lls)
    stddev_ll = 2.0 * np.std(lls) / np.sqrt(len(lls))
    print('MAR -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(mean_ll, stddev_ll))

    # Compute maximum probable explanation (MPE) queries
    mpe_data = clt.mpe(mar_data)
    assert ~np.any(np.isnan(mpe_data)), "All RVs should be assigned"

    # Evaluate the model on MPE data
    lls = clt.log_likelihood(mpe_data)
    mean_ll = np.mean(lls)
    stddev_ll = 2.0 * np.std(lls) / np.sqrt(len(lls))
    print('MPE -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(mean_ll, stddev_ll))

    # Sample some data points, using ancestral sampling (by assign all NaNs)
    # However, conditional sampling is also supported (by RVs assignments)
    samples = np.full([25, n_features], np.nan)
    samples = clt.sample(samples)
    assert ~np.any(np.isnan(samples)), "All RVs are expected to be assigned"

    # Plot the samples in a grid
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    for i in range(5):
        for j in range(5):
            x = samples[i * 5 + j].reshape(8, 8)
            axs[i, j].imshow(x, cmap='gray', vmin=0, vmax=1)
            axs[i, j].axis('off')
    fig.tight_layout()
    samples_filename = 'clt-bdigits-samples.png'
    print("Plotting generated samples to {} ...".format(samples_filename))
    fig.savefig(samples_filename)
