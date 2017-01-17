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

def normal_meanvar(data):
    """ Creates a segment cost function for a time series with a
        Normal distribution with changing mean and variance

    Args:
        data (:obj:`list` of float): 1D time series data
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    data = np.hstack(([0.0], np.array(data)))

    cumm = np.cumsum(data)
    cumm_sq = np.cumsum([val**2 for val in data])

    def cost(s, t):
        """ Cost function for normal distribution with variable variance

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        ts_i = 1.0 / (t-s)
        mu = (cumm[t] - cumm[s]) * ts_i
        sig = (cumm_sq[t] - cumm_sq[s]) * ts_i - mu**2
        sig_i = 1.0 / sig
        return (t-s) * np.log(sig) + (cumm_sq[t] - cumm_sq[s]) * sig_i - 2*(cumm[t] - cumm[s])*mu*sig_i + ((t-s)*mu**2)*sig_i

    return cost

def poisson(data):
    """ Creates a segment cost function for a time series with a
        poisson distribution with changing mean

    Args:
        data (:obj:`list` of float): 1D time series data
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    data = np.hstack(([0.0], np.array(data)))
    cumm = np.cumsum(data)

    def cost(s, t):
        """ Cost function for poisson distribution with changing mean

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        diff = cumm[t]-cumm[s]
        if diff == 0:
            return -float("Inf")
        return -2 * diff * (np.log(diff) - np.log(t-s) - 1)

    return cost

def exponential(data):
    """ Creates a segment cost function for a time series with a
        exponential distribution with changing mean

    Args:
        data (:obj:`list` of float): 1D time series data
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    data = np.hstack(([0.0], np.array(data)))
    cumm = np.cumsum(data)

    def cost(s, t):
        """ Cost function for exponential distribution with changing mean

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        return -1*(t-s) * (np.log(t-s) - np.log(cumm[t] - cumm[s]))

    return cost
