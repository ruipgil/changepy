import numpy as np

def normal_mean(data, variance):
    """ Creates a segment cost function for a time series with a
        Normal distribution with changing mean

    Args:
        data (:obj:`list` of float): 1D time series data
        variance (float): variance
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    i_variance_2 = 1 / (variance ** 2)
    cmm = [0.0]
    cmm.extend(np.cumsum(data))

    cmm2 = [0.0]
    cmm2.extend(np.cumsum(np.abs(data)))

    def cost(start, end):
        """ Cost function for normal distribution with variable mean

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        cmm2_diff = cmm2[end] - cmm2[start]
        cmm_diff = pow(cmm[end] - cmm[start], 2)
        i_diff = end - start
        diff = cmm2_diff - cmm_diff
        return (diff/i_diff) * i_variance_2

    return cost

def normal_var(data, mean):
    """ Creates a segment cost function for a time series with a
        Normal distribution with changing variance

    Args:
        data (:obj:`list` of float): 1D time series data
        variance (float): variance
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    cumm = [0.0]
    cumm.extend(np.cumsum(np.power(np.abs(data - mean), 2)))

    def cost(s, t):
        """ Cost function for normal distribution with variable variance

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        dist = float(t - s)
        diff = cumm[t] - cumm[s]
        return dist * np.log(diff/dist)

    return cost
