import numpy as np
from sklearn.datasets import load_digits

from deeprob.spn.learning.cnet_bayesian import learn_cnet_bd

if __name__ == '__main__':
    # Set up the (binarized) digits dataset
    data, _ = load_digits(return_X_y=True)
    data[data < 8] = 0
    data[data >= 8] = 1
    data = data.astype(np.float32)
    n_samples, n_features = data.shape

    # Initialize the scope and domains
    scope = list(range(n_features))

    cnet = learn_cnet_bd(data, ess=0.1, n_cand_cuts=10)

    lls = cnet.log_likelihood(data)
    mean_ll = np.mean(lls)
    stddev_ll = 2.0 * np.std(lls) / np.sqrt(len(lls))
    print('EVI -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(mean_ll, stddev_ll))
