"""
Tests to the PELT module
"""
import unittest
import numpy as np
from changepy import pelt
from changepy.costs import normal_var, normal_mean


class PELTTests(unittest.TestCase):
    """ Unittests for PELT algorithm, using the implemented cost functions
    """
    def test_pelt_normal_mean_small(self):
        """ Test normal changing mean, with smaller dataset
        """
        var = 0.1
        data = [
            0.16853651,
            0.0261112,
            -0.0655322,
            0.11575204,
            0.11388594,
            10.001775,
            9.92765733,
            10.01303474,
            9.97938986,
            10.05994745
        ]

        result = pelt(normal_mean(data, var), len(data))
        self.assertEqual(result, [0, 5])

    def test_pelt_normal_mean_big(self):
        """ Test normal changing mean, with bigger dataset
        """
        size = 100
        mean_a = 0.0
        mean_b = 10.0
        var = 0.1

        np.random.seed(19348)
        data_a = np.random.normal(mean_a, var, size)
        data_b = np.random.normal(mean_b, var, size)
        data = np.append(data_a, data_b)

        result = pelt(normal_mean(data, var), len(data))
        self.assertEqual(result, [0, size])

    def test_pelt_normal_var_big(self):
        """ Test normal changing variance, with bigger dataset
        """
        size = 100
        mean = 0.0
        var_a = 1.0
        var_b = 10.0

        np.random.seed(19348)
        data_a = np.random.normal(mean, var_a, size)
        data_b = np.random.normal(mean, var_b, size)
        data = np.append(data_a, data_b)

        result = pelt(normal_var(data, mean), len(data))
        self.assertEqual(result, [0, 100, 198])

    def test_pelt_normal_var_small(self):
        """ Test normal changing variance, with smaller dataset
        """
        data = [
            -1.82348457,
            -0.13819782,
            1.25618544,
            -0.54487136,
            -2.24769311,
            9.82204284,
            -1.0181088,
            3.93764179,
            -8.73177678,
            5.99949843
        ]

        result = pelt(normal_var(np.array(data), 0), len(data))
        self.assertEqual(result, [0, 5])

if __name__ == '__main__':
    unittest.main()
