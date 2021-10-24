import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
from deeprob.spn.learning.em import expectation_maximization
from deeprob.spn.learning.splitting.rows import split_rows_clusters
from deeprob.spn.learning.splitting.cluster import gmm

if __name__ == '__main__':
    # Setup the (binarized) digits dataset
    x_data, y_data = load_digits(return_X_y=True)
    targets = np.unique(y_data)
    n_targets = len(targets)
    x_data[x_data < 8] = 0
    x_data[x_data >= 8] = 1
    x_data = x_data.astype(np.float32)
    n_samples, n_features = x_data.shape

    # Build the complete training data, i.e. features + targets
    data_train = np.column_stack([x_data, y_data[:, np.newaxis]])

    # Initialize the SPN scope and leaf distributions and domains
    scope = list(range(n_features + 1))
    distributions = [spn.Bernoulli] * n_features + [spn.Categorical]
    domains = [[0, 1]] * n_features + [list(range(len(targets)))]

    # Instantiate the random state
    random_state = np.random.RandomState(42)

    # Build a SPN-based classifier using mixtures of Binary Chow-Liu Trees (CLTs)
    weights = list()
    children = list()
    for t in targets:  # Consider the data of each class
        mask = y_data == t
        loc_data, loc_targets = x_data[mask], y_data[mask]
        weights.append(len(loc_data) / n_samples)

        # Cluster the data using a Gaussian Mixture Model (GMM) with two components
        clusters = gmm(loc_data, distributions[:-1], domains[:-1], random_state, n=2)
        data_slices, weights_slices = split_rows_clusters(loc_data, clusters)

        # Build a mixture of two Binary CLTs
        # Use alpha=0.01 as smoothing factor
        mixture_node = spn.Sum(
            children=[spn.BinaryCLT(scope[:-1]), spn.BinaryCLT(scope[:-1])],
            weights=weights_slices
        )
        for d, clt in zip(data_slices, mixture_node.children):
            clt.fit(d, domains[:-1], alpha=0.01, random_state=random_state)

        # Build a product node by integrating the target categorical distribution
        target_node = spn.Categorical(n_features)
        target_node.fit(loc_targets, domains[-1], alpha=0.01)
        product_node = spn.Product(children=[target_node, mixture_node])

        # Append the resulting product node
        children.append(product_node)

    # Initialize the SPN root node as a mixture of SPNs, one for each class
    root = spn.Sum(children=children, weights=weights)

    # Initialize the node IDs of the SPN
    spn.assign_ids(root)

    # Evaluate the model by computing the average log-likelihood with two standard deviations
    lls = spnalg.log_likelihood(root, data_train)
    mean_ll = np.mean(lls)
    stddev_ll = 2.0 * np.std(lls) / np.sqrt(len(lls))
    print('Base -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(mean_ll, stddev_ll))

    # Run batch Expectation Maximization (EM), but keeping the starting parameters
    expectation_maximization(
        root, data_train, num_iter=250, batch_perc=0.5, step_size=0.1,
        random_init=False, random_state=random_state
    )

    # Evaluate the model after EM
    lls = spnalg.log_likelihood(root, data_train)
    mean_ll = np.mean(lls)
    stddev_ll = 2.0 * np.std(lls) / np.sqrt(len(lls))
    print('After EM -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(mean_ll, stddev_ll))

    # Sample some data points, using conditional sampling (by assign all NaNs except the targets)
    nan_features = np.full([5 * n_targets, n_features], np.nan)
    all_targets = np.tile(targets, reps=5)
    samples = np.column_stack([nan_features, all_targets])
    samples = spnalg.sample(root, samples)[:, :-1]
    assert ~np.any(np.isnan(samples)), "All RVs should be assigned"

    # Plot the samples in a grid
    fig, axs = plt.subplots(5, n_targets, figsize=(n_targets, 5))
    for i in range(5):
        for j in range(n_targets):
            x = samples[i * n_targets + j].reshape(8, 8)
            axs[i, j].imshow(x, cmap='gray', vmin=0, vmax=1)
            axs[i, j].axis('off')
    fig.tight_layout()
    samples_filename = 'spn-clt-bdigits-samples.png'
    print("Plotting generated samples to {} ...".format(samples_filename))
    fig.savefig(samples_filename)

    # Save the SPN model (structure and parameters)
    spn_filename = 'spn-clt-bdigits.json'
    print("Saving the SPN structure and parameters to {} ...".format(spn_filename))
    spn.save_spn_json(root, spn_filename)
