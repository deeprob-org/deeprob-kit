import numpy as np
from sklearn.datasets import load_boston

import deeprob.spn.structure as spn

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

    # Plot the CLT
    clt_filename = 'clt-bboston.svg'
    print("Plotting the learnt CLT to {} ...".format(clt_filename))
    spn.plot_binary_clt(clt, clt_filename)
