# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Type, List

import numpy as np
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, DensityMixin, ClassifierMixin

from deeprob.spn.structure.leaf import Leaf, Bernoulli, Categorical
from deeprob.spn.learning.wrappers import learn_estimator, learn_classifier
from deeprob.spn.algorithms.inference import log_likelihood, mpe
from deeprob.spn.algorithms.sampling import sample


class SPNEstimator(BaseEstimator, DensityMixin):
    def __init__(
        self,
        distributions: List[Type[Leaf]],
        domains: Optional[List[Union[list, tuple]]] = None,
        **kwargs
    ):
        """
        Scikit-learn density estimator model for Sum Product Networks (SPNs).

        :param distributions: A list of distribution classes (one for each feature).
        :param domains: A list of domains (one for each feature).
        :param kwargs: Additional arguments to pass to the SPN learner.
        """
        super().__init__()
        self.distributions = distributions
        self.domains = domains
        self.kwargs = kwargs
        self.spn_ = None
        self.n_features_ = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the SPN density estimator.

        :param X: The training data.
        :param y: Ignored, only for scikit-learn API convention.
        :return: Itself.
        """
        self.spn_ = learn_estimator(X, self.distributions, self.domains, **self.kwargs)
        self.n_features_ = X.shape[1]
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the SPN density estimator, i.e. compute the log-likelihood.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The log-likelihood of the inputs.
        """
        return log_likelihood(self.spn_, X)

    def mpe(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the un-observed variable by maximum at posterior estimation (MPE).

        :param X: The inputs having some NaN values.
        :return: The MPE assignment to un-observed variables.
        """
        return mpe(self.spn_, X, inplace=False)

    def sample(self, n: Optional[int] = None, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample from the modeled distribution.

        :param n: The number of samples. It must be None if X is not None. If None, n=1 is assumed.
        :param X: Data used for conditional sampling. It can be None for full sampling.
        :return: The samples.
        :raise ValueError: If both parameters 'n' and 'X' are passed by.
        """
        if n is not None and X is not None:
            raise ValueError("Only one between 'n' and 'X' can be specified")

        if X is not None:
            # Conditional sampling
            return sample(self.spn_, X, inplace=False)
        else:
            # Full sampling
            n = 1 if n is None else n
            x = np.tile(np.nan, [n, self.n_features_])
            return sample(self.spn_, x, inplace=True)

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> dict:
        """
        Return the mean log-likelihood and two standard deviations on the given test data.

        :param X: The inputs. They can be marginalized using NaNs.
        :param y: Ignored. Specified only for scikit-learn API compatibility.
        :return: A dictionary consisting of two keys "mean_ll" and "stddev_ll",
                 representing respectively the mean log-likelihood and two standard deviations.
        """
        ll = self.predict_log_proba(X)
        mean_ll = np.mean(ll)
        stddev_ll = np.std(ll)
        return {
            'mean_ll': mean_ll,
            'stddev_ll': 2.0 * stddev_ll / np.sqrt(len(X))
        }


class SPNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        distributions: List[Type[Leaf]],
        domains: Optional[List[Union[list, tuple]]] = None,
        **kwargs
    ):
        """
        Scikit-learn classifier model for Sum Product Networks (SPNs).

        :param distributions: A list of distribution classes (one for each feature).
        :param domains: A list of domains (one for each feature).
        :param kwargs: Additional arguments to pass to the SPN learner.
        """
        super().__init__()
        self.distributions = distributions
        self.domains = domains
        self.kwargs = kwargs
        self.spn_ = None
        self.n_features_ = 0
        self.n_classes_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SPN density estimator.

        :param X: The training data.
        :param y: The data labels.
        :return: Itself.
        """
        # Build the training data, consisting of labels
        y = np.expand_dims(y, axis=1)
        data = np.hstack([X, y])

        # Constructs the list of distributions
        n_classes = len(np.unique(y))
        if n_classes == 2:
            # Use bernoulli for binary classification
            distributions = self.distributions + [Bernoulli]
        else:
            # otherwise, use a categorical distribution
            distributions = self.distributions + [Categorical]

        self.spn_ = learn_classifier(data, distributions, self.domains, **self.kwargs)
        self.n_features_ = X.shape[1]
        self.n_classes_ = n_classes
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the SPN classifier.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The predicted classes.
        """
        # Build the testing data, having X as features assignments and NaNs for labels
        data = np.hstack([X, np.full([len(X), 1], np.nan)])

        # Make classification using maximum probable explanation (MPE)
        mpe(self.spn_, data, inplace=True)

        # Return the classifications for each sample
        return data[:, -1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the SPN classifier, using probabilities.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The prediction probabilities for each class.
        """
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the SPN classifier, using log-probabilities.

        :param X: The inputs. They can be marginalized using NaNs.
        :return: The prediction log-probabilities for each class.
        """
        # Build the testing data, having X as features assignments and NaNs for labels
        data = np.hstack([X, np.tile(np.nan, [len(X), 1])])

        # Make probabilistic classification by computing the log-likelihoods at sub-class SPN
        _, lls = log_likelihood(self.spn_, data, return_results=True)

        # Collect the predicted class probabilities
        class_ids = [c.id for c in self.spn_.children]
        class_ll = np.log(self.spn_.weights) + lls[class_ids]
        return log_softmax(class_ll, axis=1)

    def sample(self, n: Optional[int] = None, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample from the modeled conditional distribution.

        :param n: The number of samples. It must be None if y is not None. If None, n=1 is assumed.
        :param y: Labels used for conditional sampling. It can be None for un-conditional sampling.
        :return: The samples.
        """
        if n is not None and y is not None:
            raise ValueError("Only one between 'n' and 'y' can be specified")

        # Conditional sampling
        if y is not None:
            y = np.expand_dims(y, axis=1)
            x = np.hstack([np.tile(np.nan, [len(y), self.n_features_]), y])
            return sample(self.spn_, x, inplace=False)

        # Full sampling
        n = 1 if n is None else n
        x = np.tile(np.nan, [n, self.n_features_ + 1])
        return sample(self.spn_, x, inplace=True)
