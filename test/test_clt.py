import unittest
import tempfile
import numpy as np

from itertools import product
from experiments.datasets import load_binary_dataset

from deeprob.spn.utils.validity import is_structured_decomposable
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.io import save_binary_clt_json, load_binary_clt_json
from deeprob.spn.algorithms.inference import log_likelihood


class TestCLT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCLT, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        random_state = np.random.RandomState(42)
        data, _, _ = load_binary_dataset('experiments/datasets', 'nltcs', raw=True)
        data = data.astype(np.float32)
        cls.n_samples, cls.n_features = data.shape
        cls.evi_data = data[random_state.choice(len(data), size=5000)]
        cls.mar_data = cls.evi_data.copy()
        cls.mar_data[random_state.rand(*cls.mar_data.shape) < 0.2] = np.nan
        cls.complete_data = np.array([list(i) for i in product([0, 1], repeat=cls.n_features)], dtype=np.float32)

    def __learn_binary_clt(self):
        scope = list(range(self.n_features))
        clt = BinaryCLT(scope)
        clt.fit(self.evi_data, [[0, 1]] * self.n_features, alpha=0.1, random_state=42)
        return clt

    def test_complete_inference(self):
        clt = self.__learn_binary_clt()
        ls = clt.likelihood(self.complete_data)
        lls = clt.log_likelihood(self.complete_data)
        self.assertAlmostEqual(np.sum(ls).item(), 1.0, places=6)
        self.assertAlmostEqual(np.sum(np.exp(lls)).item(), 1.0, places=6)

    def test_mar_inference(self):
        clt = self.__learn_binary_clt()
        evi_ll = clt.log_likelihood(self.evi_data).mean()
        mar_ll = clt.log_likelihood(self.mar_data).mean()
        self.assertGreater(mar_ll, evi_ll)

    def test_mpe_inference(self):
        clt = self.__learn_binary_clt()
        evi_ll = clt.log_likelihood(self.evi_data).mean()
        mpe_data = clt.mpe(self.mar_data)
        mpe_ll = clt.log_likelihood(mpe_data).mean()
        self.assertFalse(np.any(np.isnan(mpe_data)))
        self.assertGreater(mpe_ll, evi_ll)

    def test_pc_conversion(self):
        clt = self.__learn_binary_clt()
        clt_evi_ll = clt.log_likelihood(self.evi_data).mean()
        clt_mar_ll = clt.log_likelihood(self.mar_data).mean()
        spn = clt.to_pc()
        spn_evi_ll = log_likelihood(spn, self.evi_data).mean()
        spn_mar_ll = log_likelihood(spn, self.mar_data).mean()
        self.assertAlmostEqual(clt_evi_ll, spn_evi_ll, places=6)
        self.assertAlmostEqual(clt_mar_ll, spn_mar_ll, places=6)
        self.assertIsNone(is_structured_decomposable(spn))

    def test_save_load_json(self):
        clt = self.__learn_binary_clt()
        ll = clt.log_likelihood(self.evi_data).mean()
        with tempfile.TemporaryFile('r+') as f:
            save_binary_clt_json(clt, f)
            f.seek(0)
            loaded_clt = load_binary_clt_json(f)
        loaded_ll = loaded_clt.log_likelihood(self.evi_data).mean()
        self.assertEqual(ll, loaded_ll)


if __name__ == '__main__':
    unittest.main()
