import unittest
import numpy as np

from deeprob.utils.random import check_random_state


class TestRandom(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRandom, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.seed = 42

    def test_check_random_state(self):
        self.assertEqual(check_random_state().__class__, np.random.RandomState)
        self.assertEqual(check_random_state(self.seed).randint(5), 3)
        self.assertEqual(check_random_state(np.random.RandomState(42)).randint(5), 3)
        self.assertRaises(ValueError, check_random_state, np.arange(10))


if __name__ == '__main__':
    unittest.main()
