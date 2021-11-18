import unittest
import tempfile

from sklearn.datasets import load_diabetes
from test.utils import *

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
        data, _, = load_diabetes(return_X_y=True)
        data = (data < np.median(data, axis=0)).astype(np.float32)
        cls.root_id = 1
        cls.n_samples, cls.n_features = data.shape
        cls.evi_data = resample_data(data, 1000, random_state)
        cls.mar_data = random_marginalize_data(cls.evi_data, 0.2, random_state)

        cls.complete_data = complete_binary_data(cls.n_features)
        mar_features = [7, cls.root_id, 5, 9]
        cls.complete_mar_data = complete_marginalized_binary_data(cls.n_features, mar_features)
        cls.complete_mpe_data = complete_posterior_binary_data(cls.n_features, mar_features)

        cls.approx_iter = 250

    def __learn_binary_clt(self):
        scope = list(range(self.n_features))
        clt = BinaryCLT(scope, root=self.root_id)
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

    def test_mpe_complete_inference(self):
        clt = self.__learn_binary_clt()
        complete_lls = clt.log_likelihood(self.complete_data)
        mpe_data = clt.mpe(self.complete_mar_data)
        mpe_ids = binary_data_ids(mpe_data).tolist()
        expected_mpe_ids = compute_mpe_ids(self.complete_mpe_data, complete_lls.squeeze())
        self.assertEqual(mpe_ids, expected_mpe_ids)

    def test_ancestral_sampling(self):
        clt = self.__learn_binary_clt()
        evi_ll = clt.log_likelihood(self.evi_data).mean()
        np.random.seed(42)
        samples = np.empty(shape=(10000, self.n_features), dtype=np.float32)
        approx_lls = list()
        for _ in range(self.approx_iter):
            samples[:] = np.nan
            samples = clt.sample(samples)
            approx_lls.extend(clt.log_likelihood(samples).squeeze().tolist())
        approx_ll = np.mean(approx_lls).item()
        self.assertAlmostEqual(evi_ll, approx_ll, places=2)

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
