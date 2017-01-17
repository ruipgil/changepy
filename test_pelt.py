"""
Tests to the PELT module
"""
import unittest
import numpy as np
from changepy import pelt
from changepy.costs import normal_var, normal_mean, normal_meanvar, exponential, poisson
import matplotlib.pyplot as plt




class PELTTests(unittest.TestCase):
    """ Unittests for PELT algorithm, using the implemented cost functions
    """
    # def test_pelt_normal_mean_small(self):
    #     """ Test normal changing mean, with smaller dataset
    #     """
    #     var = 0.1
    #     data = [
    #         0.16853651,
    #         0.0261112,
    #         -0.0655322,
    #         0.11575204,
    #         0.11388594,
    #         10.001775,
    #         9.92765733,
    #         10.01303474,
    #         9.97938986,
    #         10.05994745
    #     ]
    #
    #     result = pelt(normal_mean(data, var), len(data))
    #     self.assertEqual(result, [0, 5])
    #
    # def test_pelt_normal_mean_big(self):
    #     """ Test normal changing mean, with bigger dataset
    #     """
    #     size = 100
    #     mean_a = 0.0
    #     mean_b = 10.0
    #     var = 0.1
    #
    #     np.random.seed(19348)
    #     data_a = np.random.normal(mean_a, var, size)
    #     data_b = np.random.normal(mean_b, var, size)
    #     data = np.append(data_a, data_b)
    #
    #     result = pelt(normal_mean(data, var), len(data))
    #     self.assertEqual(result, [0, size])
    #
    # def test_pelt_normal_var_big(self):
    #     """ Test normal changing variance, with bigger dataset
    #     """
    #     size = 100
    #     mean = 0.0
    #     var_a = 1.0
    #     var_b = 10.0
    #
    #     np.random.seed(19348)
    #     data_a = np.random.normal(mean, var_a, size)
    #     data_b = np.random.normal(mean, var_b, size)
    #     data = np.append(data_a, data_b)
    #
    #     result = pelt(normal_var(data, mean), len(data))
    #     self.assertEqual(result, [0, 100, 198])
    #
    # def test_pelt_normal_var_small(self):
    #     """ Test normal changing variance, with smaller dataset
    #     """
    #     data = [
    #         -1.82348457,
    #         -0.13819782,
    #         1.25618544,
    #         -0.54487136,
    #         -2.24769311,
    #         9.82204284,
    #         -1.0181088,
    #         3.93764179,
    #         -8.73177678,
    #         5.99949843
    #     ]
    #
    #     result = pelt(normal_var(np.array(data), 0), len(data))
    #     self.assertEqual(result, [0, 5])
    #
    # def test_pelt_normal_meanvar_big(self):
    #     np.random.seed(1)
    #     data = np.hstack((np.random.normal(0, 1, 100), np.random.normal(10, 10, 100)))
    #     cost = normal_meanvar(data)
    #     result = pelt(cost, len(data))
    #     self.assertEqual(result, [0, 1, 3, 12, 14, 21, 25, 32, 35, 41, 43, 48, 51, 53, 55, 58, 66, 69, 71, 77, 80, 82, 86, 89, 91, 100, 139, 142, 145, 147, 150, 152, 155, 158, 162, 168, 173, 176, 187, 189, 196, 198])

    def test_pelt_exponential_big(self):
        np.random.seed(1)
        data = np.hstack((np.random.exponential(1, 100), np.random.exponential(2.1, 100)))
        cost = exponential(data)
        result = pelt(cost, len(data))
        self.assertEqual(result, [0, 101])

    def test_poisson_r(self):
        data = [4,4,3,8,4,3,9,6,5,4,6,5,3,4,6,0,3,5,4,4,7,7,5,2,6,4,8,6,3,3,2,1,1,5,9,7,3,3,6,4,6,2,4,3,6,3,10,6,9,3,8,2,5,8,5,4,4,3,1,5,9,5,4,2,7,4,0,1,2,3,6,4,6,2,8,6,2,5,6,4,6,6,2,5,5,7,11,7,6,9,7,9,4,4,7,5,8,5,8,9,12,6,1,5,6,6,3,7,7,8,10,4,5,8,2,7,8,7,10,9,7,4,3,8,6,7,4,7,4,11,12,6,10,5,8,6,8,5,4,8,7,5,9,7,8,6,9,8,5,6,13,4,8,5,11,4,8,5,5,8,7,10,8,8,4,5,4,4,11,8,5,10,4,4,8,9,5,5,12,1,4,8,4,6,6,9,3,6,7,8,3,3,6,7,5,7,5,4,10,5,5,6,7,4,3,3,3,6,9,7,9,6,6,7,5,6,6,7,7,6,12,10,6,5,10,10,9,8,19,24,16,25,15,24,16,21,18,16,21,9,11,18,22,16,21,18,11,19,20,15,17,25,16,19,22,20,17,25,27,12,20,23,20,22,19,24,21,19,19,20,18,21,14,16,15,18,24,19,17,16,21,20,22,14,21,21,16,16,12,15,22,21,18,10,21,16,22,14,22,16,19,20,26,21,20,23,6,21,17,18,24,18,18,14,17,16,16,19,16,19,15,18,19,22,23,22,19,16,21,14,24,22,19,21,16,16,15,21,23,17,15,21,16,20,13,17,16,20,9,21,17,23,15,20,11,20,15,20,19,18,5,6,9,4,8,6,5,8,4,5,6,7,7,8,4,3,7,3,2,8,5,7,4,12,7,8,7,1,6,8,5,9,9,9,6,6,6,4,9,4,7,2,5,7,11,4,4,9,6,4,10,11,11,6,12,5,5,6,5,3,3,6,13,3,4,4,7,6,8,5,9,6,7,11,2,6,9,5,3,3,3,6,1,2,4,1,5,1,3,6,3,2,4,2,3,3,3,0,1,7,6,1,3,2,3,1,7,1,2,4,4,3,4,5,2,3,2,5,1,0,4,5,0,4,4,3,3,2,2,2,1,4,1,4,5,2,2,6,3,2]
        cost = poisson(data)
        result = pelt(cost, len(data), 2*np.log(len(data)))
        self.assertEqual(result, [0, 85, 228, 360, 438])

if __name__ == '__main__':
    unittest.main()
