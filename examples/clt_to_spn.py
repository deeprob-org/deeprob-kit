import numpy as np
from sklearn.datasets import load_boston

import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
import deeprob.spn.utils as spnutils

if __name__ == '__main__':
    # Load the boston dataset and binarize it
    data, _ = load_boston(return_X_y=True)
    avg_features = np.mean(data, axis=0)
    data = (data < avg_features).astype(np.float32)
    n_samples, n_features = data.shape

    # Instantiate the random state
    random_state = np.random.RandomState(42)

    # Fit a binary CLT
    scope = list(range(n_features))
    domain = [[0, 1]] * n_features
    clt = spn.BinaryCLT(scope)
    clt.fit(data, domain, alpha=0.1, random_state=random_state)

    # Evaluate the binary CLT
    clt_ll = clt.log_likelihood(data).mean()

    # Convert the CLT into a structured decomposable SPN
    root = clt.to_pc()
    spnutils.check_spn(root, labeled=True, smooth=True, decomposable=True, structured_decomposable=True)

    # Plot the SPN
    spn_filename = 'clt-to-spn.svg'
    print("Plotting the compiled SPN to {} ...".format(spn_filename))
    spn.plot_spn(root, spn_filename)

    # Evaluate the SPN
    spn_ll = spnalg.log_likelihood(root, data).mean()

    # Note that the SPN should encode the same probability distribution
    assert np.isclose(clt_ll, spn_ll), "CLT to SPN conversion failed"
