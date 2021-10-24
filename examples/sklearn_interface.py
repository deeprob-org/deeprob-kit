import numpy as np
from sklearn.datasets import load_breast_cancer

import deeprob.spn.structure as spn
from deeprob.spn.models.sklearn import SPNEstimator, SPNClassifier

if __name__ == '__main__':
    # Load the dataset and set the features distributions
    data, target = load_breast_cancer(return_X_y=True)
    _, n_features = data.shape
    distributions = [spn.Gaussian] * n_features

    # =================================================================================================================

    # Instantiate and fit a SPN density estimator
    clf = SPNEstimator(
        distributions,
        learn_leaf='mle',     # Learn leaf distributions by MLE
        split_rows='kmeans',  # Use K-Means for splitting rows
        split_cols='rdc',     # Use RDC for splitting columns
        min_rows_slice=64,    # The minimum number of rows required to split furthermore
        random_state=42       # The random state, used for reproducibility
    )
    clf.fit(data)

    # Compute the average log-likelihood and two standard deviations
    score = clf.score(data)
    print('Train data -- Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(score['mean_ll'], score['stddev_ll']))

    # Sample some data and compute the average log-likelihood and two standard deviations
    samples = clf.sample(n=100)
    score = clf.score(samples)
    print('Sampled data - Mean LL: {:.4f} - Stddev LL: {:.4f}'.format(score['mean_ll'], score['stddev_ll']))

    # =================================================================================================================

    # Instantiate and fit a SPN classifier
    clf = SPNClassifier(
        distributions,
        learn_leaf='mle',     # Learn leaf distributions by MLE
        split_rows='kmeans',  # Use K-Means for splitting rows
        split_cols='rdc',     # Use RDC for splitting columns
        min_rows_slice=64,    # The minimum number of rows required to split furthermore
        random_state=42       # The random state, used for reproducibility
    )
    clf.fit(data, target)

    # Compute the accuracy score
    print('Train data -- Accuracy: {:.2f}'.format(clf.score(data, target)))

    # Sample some data from the conditional distribution and compute the accuracy score
    classes = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    samples = clf.sample(y=classes)
    print('Sampled data -- Accuracy: {:.2f}'.format(clf.score(samples[:, :-1], classes)))
