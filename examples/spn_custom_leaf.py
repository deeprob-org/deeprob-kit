import numpy as np
import scipy.stats as ss

import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
import deeprob.spn.utils as spnutils
from deeprob.spn.learning import learn_spn


class Cauchy(spn.Leaf):
    LEAF_TYPE = spn.LeafType.CONTINUOUS

    def __init__(self, scope: int, loc: float = 0.0, scale: float = 1.0):
        super().__init__(scope)
        self.loc = loc
        self.scale = scale

    def fit(self, data: np.ndarray, domain: tuple, **kwargs):
        self.loc, self.scale = ss.cauchy.fit(data)

    def em_init(self, random_state: np.random.RandomState):
        raise NotImplemented("EM parameters initialization not yet implemented for Cauchy distributions")

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        raise NotImplemented("EM step not yet implemented for Cauchy distributions")

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        ls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ls[~mask] = ss.cauchy.pdf(x[~mask], loc=self.loc, scale=self.scale)
        return ls

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        lls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        lls[~mask] = ss.cauchy.logpdf(x[~mask], loc=self.loc, scale=self.scale)
        return lls

    def mpe(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = self.loc
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = ss.cauchy.rvs(loc=self.loc, scale=self.scale, size=np.count_nonzero(mask))
        return x

    def moment(self, k: int = 1) -> float:
        return ss.cauchy.moment(k)

    def params_count(self) -> int:
        return 2

    def params_dict(self) -> dict:
        return {'loc': self.loc, 'scale': self.scale}


if __name__ == '__main__':
    # Sample some random data
    random_state = np.random.RandomState(42)
    n_samples, n_features = 1000, 4
    data = random_state.randn(n_samples, n_features)

    # Learn a SPN from data using Cauchy distributions at leaves
    distributions = [Cauchy] * n_features
    domains = [(-9.0, 9.0)] * n_features
    root = learn_spn(
        data, distributions, domains,
        learn_leaf='mle',          # The MLE learn leaf method will use the fit() method of leaf's class
        random_state=random_state  # Set the random state manually
    )

    # Compute the average likelihood
    ls = spnalg.likelihood(root, data)
    print("Average Likelihood: {:.4f}".format(np.mean(ls)))

    # Print some statistics about the model's structure and parameters
    print("SPN structure and parameters statistics:")
    print(spnutils.compute_statistics(root))

    # Save the model to a JSON file
    spn_filename = 'spn-custom-cauchy.json'
    print("Saving the SPN structure and parameters to {} ...".format(spn_filename))
    spn.save_spn_json(root, spn_filename)
    del root

    # Reload the model from file
    # Note that we need to specify the custom leaf
    print("Re-loading the SPN structure and parameters from {} ...".format(spn_filename))
    root = spn.load_spn_json('spn-custom-cauchy.json', leaves=[Cauchy])
    ls = spnalg.likelihood(root, data)
    print("Average Likelihood: {:.4f}".format(np.mean(ls)))
