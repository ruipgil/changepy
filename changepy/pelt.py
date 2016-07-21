import numpy as np

def pelt(cost, length, pen=None):
    """ PELT algorithm to compute changepoints in time series

    Ported from:
        https://github.com/STOR-i/Changepoints.jl
        https://github.com/rkillick/changepoint/
    Reference:
        Killick R, Fearnhead P, Eckley IA (2012) Optimal detection
            of changepoints with a linear computational cost, JASA
            107(500), 1590-1598

    Args:
        cost (function): cost function, with the following signature,
            (int, int) -> float
            where the parameters are the start index, and the second
            the last index of the segment to compute the cost.
        length (int): Data size
        pen (float, optional): defaults to log(n)
    Returns:
        (:obj:`list` of int): List with the indexes of changepoints
    """
    if pen is None:
        pen = np.log(length)

    optimal_costs = np.zeros(length + 1)
    optimal_costs[0] = -pen

    changepoints = np.zeros(length, dtype=np.int)

    candidates = [0]

    for i in range(1, length):
        costs = []
        for candidate in candidates:
            ccost = cost(candidate, i + 1)
            costs.append(ccost)

        sel = optimal_costs[np.array(candidates, dtype=np.int)]
        temp = np.add(sel, costs)

        # Since pen is a constant, we don't need to add it to all
        # the element of the array, only to the min value
        min_val = min(temp) + pen
        min_idx = np.argmin(temp)

        optimal_costs[i+1] = min_val
        changepoints[i] = candidates[min_idx]

        ineq_prune = [val < min_val for val in temp]
        _candidates = []
        for j, val in enumerate(ineq_prune):
            if val:
                _candidates.append(candidates[j])
        candidates = _candidates

        candidates.append(i)

    last = changepoints[-1]
    backtrack = [last]
    while last > 0:
        last = changepoints[last-1]
        backtrack.append(last)

    return sorted(backtrack)
