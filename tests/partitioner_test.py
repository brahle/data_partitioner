"""Unit test for the utils module."""

import unittest
import unittest.mock

from data_partitioner.partitioner import (
    LinearFakeRandomFunction,
    partitioner,
    pseudorandom_function,
)

class TesLinearFakeRandomFunction(unittest.TestCase):
    """A test class for the utils module."""

    def test_limit_10(self):
        """Test with the number of elements set to 10"""
        limit = 10
        rand = LinearFakeRandomFunction(limit)
        for i in range(0, limit):
            self.assertAlmostEqual(rand(i), i/limit)

    def test_limit_0(self):
        """Checks the class doesn't accept an invalid limit."""
        with self.assertRaises(AssertionError):
            LinearFakeRandomFunction(0)

    def test_index_larger_than_limit(self):
        """Checks the index that's larger than limit is causing a failure."""
        rand = LinearFakeRandomFunction(2)
        with self.assertRaises(AssertionError):
            rand(3)

    def test_negative_index(self):
        """Checks that a negative index is causing a failure."""
        rand = LinearFakeRandomFunction(2)
        with self.assertRaises(AssertionError):
            rand(-3)

class TestPseudoRandomFunction(unittest.TestCase):
    """A test for the pseudorandom function."""

    def test_random_function_1(self):
        """Tests if pseudorandom_function is using random.random()"""
        def fake_random():
            """Fake random function"""
            return 0.5
        with unittest.mock.patch("random.random", fake_random):
            for i in range(0, 100):
                self.assertEqual(pseudorandom_function(i), 0.5)

    def test_random_function_2(self):
        """Tests if pseudorandom_function is using random.random()"""
        def fake_random():
            """Fake random function"""
            return 0.75
        with unittest.mock.patch("random.random", fake_random):
            for i in range(0, 100):
                self.assertEqual(pseudorandom_function(i), 0.75)

    def test_negative_index(self):
        """Checks that a negative index is causing a failure."""
        with self.assertRaises(AssertionError):
            pseudorandom_function(-3)

class TestPartitioner(unittest.TestCase):
    """A test for the partitioner."""

    def test_split_in_two(self):
        """Tests a simple linear split into two pieces."""
        i = 0
        for partition in partitioner(10, [1, 1], LinearFakeRandomFunction(10)):
            self.assertEqual(i // 5, partition)
            i += 1

    def test_split_in_three(self):
        """Tests a simple linear split into three equal pieces."""
        i = 0
        expected = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        for partition in partitioner(10, [1, 1, 1], LinearFakeRandomFunction(10)):
            self.assertEqual(expected[i], partition)
            i += 1

    def test_split_in_three_with_empty(self):
        """Tests a simple linear split into three equal pieces."""
        i = 0
        expected = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
        for partition in partitioner(10, [1, 0, 1], LinearFakeRandomFunction(10)):
            self.assertEqual(expected[i], partition)
            i += 1

    def test_split_in_three_unequal(self):
        """Tests a simple linear split into three unequal pieces."""
        i = 0
        expected = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        for partition in partitioner(10, [1, 1, 2], LinearFakeRandomFunction(10)):
            self.assertEqual(expected[i], partition)
            i += 1

    def test_negative_weight(self):
        """Assert that using a weight thats' not positive is a failure."""
        with self.assertRaises(AssertionError):
            partitioner(10, [-1, 1]).__next__()

    def test_weight_sum_0(self):
        """Assert that using a weight thats' not positive is a failure."""
        with self.assertRaises(AssertionError):
            partitioner(10, [0, 0]).__next__()

    def test_non_positive_n_of_elements(self):
        """Assert that using a weight thats' not positive is a failure."""
        with self.assertRaises(AssertionError):
            partitioner(0, [1, 1]).__next__()

if __name__ == "__main__":
    unittest.main()
