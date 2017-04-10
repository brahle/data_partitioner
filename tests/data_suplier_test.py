"""
Tests for the data suplier.
"""

import unittest

from data_partitioner.data_suplier import DataSuplier

class DataSuplierTest(unittest.TestCase):
    """A class that tests data suplier class"""

    def setUp(self):
        self.dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def test_iteration(self):
        """Tests that one can iterate over a dataset suplier .test_set and training_set methods."""
        suplier = DataSuplier(self.dataset)
        len_dataset = len(self.dataset)
        test = []
        training = []
        for element in suplier.test_set():
            self.assertGreaterEqual(element[0], 0)
            self.assertLess(element[0], len_dataset)
            self.assertIn(element[1], self.dataset)
            self.assertNotIn(element, test)
            test.append(element)
        for element in suplier.training_set():
            self.assertGreaterEqual(element[0], 0)
            self.assertLess(element[0], len_dataset)
            self.assertIn(element[1], self.dataset)
            self.assertNotIn(element, test)
            self.assertNotIn(element, training)
            training.append(element)
        self.assertEqual(len(test) + len(training), len_dataset)

    def test_stability(self):
        """Tests the stability of the suplier."""
        suplier = DataSuplier(self.dataset)
        test = list(suplier.test_set())
        training = list(suplier.training_set())
        for _ in range(100):
            self.assertListEqual(test, list(suplier.test_set()))
            self.assertListEqual(training, list(suplier.training_set()))

    def test_training_percent(self):
        """Tests that the setting the training percent affects the distributions."""
        self._check_distribution(DataSuplier(list(range(10000))), 0.2)
        self._check_distribution(DataSuplier(list(range(10000)), 0.5), 0.5)

    def _check_distribution(self, suplier, test_percent, epsilon=0.05):
        test_n = len(list(suplier.test_set()))
        training_n = len(list(suplier.training_set()))
        total_n = test_n + training_n
        self.assertAlmostEqual(test_n / total_n, test_percent, delta=epsilon*test_n/total_n)
        self.assertAlmostEqual(
            training_n / total_n,
            1 - test_percent,
            delta=epsilon*training_n/total_n
        )

    def test_partitioner(self):
        """Tests ifi the partitioner that is suplied in constructor is actually called."""
        suplier = DataSuplier(self.dataset, partitioning_function=mock_partitioner)
        for element in suplier.training_set():
            self.assertEqual(element[0] % 2, 0)
        for element in suplier.test_set():
            self.assertEqual(element[0] % 2, 1)

def mock_partitioner(i):
    """Partitioning function that will partition elements based on them being odd or even."""
    return i % 2

if __name__ == "__main__":
    unittest.main()
