import numpy as np

import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
import deeprob.spn.utils as spnutils
from deeprob.spn.learning.leaf import learn_mle, learn_naive_factorization

if __name__ == '__main__':
    # Sample some binary data randomly
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    data = np.random.binomial(1, p=0.4, size=[n_samples, n_features])

    # Set the features distributions and domains
    distributions = [spn.Bernoulli] * n_features
    domains = [[0, 1]] * n_features  # Use lists to specify discrete domains

    # Learn a naive factorized model from a subset of the data
    scope = [5, 1, 7]
    dists = [distributions[s] for s in scope]
    doms = [domains[s] for s in scope]
    naive = learn_naive_factorization(
        data[:, scope], dists, doms, scope,
        learn_leaf_func=learn_mle,  # Use MLE to learn the leaf distributions
        alpha=0.01  # Additional learn_mle parameters, for example the Laplace smoothing factor
    )

    # Compute the average likelihood
    ls = spnalg.likelihood(naive, data)
    print("Average Likelihood: {:.4f}".format(np.mean(ls)))

    # Print some statistics about the model's structure and parameters
    print("SPN structure and parameters statistics:")
    print(spnutils.compute_statistics(naive))
