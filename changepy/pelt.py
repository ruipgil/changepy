import numpy as np

def find_min(arr, val=0.0):
    """ Finds the minimum value and index

    Args:
        arr (np.array)
        val (float, optional): value to add
    Returns:
        (float, int): minimum value and index
    """
    return min(arr) + val, np.argmin(arr)

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

    F = np.zeros(length + 1)
    R = np.array([0], dtype=np.int)
    candidates = np.zeros(length + 1, dtype=np.int)

    F[0] = -pen

    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = np.zeros(len(cpt_cands))
        for i in range(0, len(cpt_cands)):
            seg_costs[i] = cost(cpt_cands[i], tstar)

        F_cost = F[cpt_cands] + seg_costs
        F[tstar], tau = find_min(F_cost, pen)
        candidates[tstar] = cpt_cands[tau]

        ineq_prune = [val < F[tstar] for val in F_cost]
        R = [cpt_cands[j] for j, val in enumerate(ineq_prune) if val]
        R.append(tstar - 1)
        R = np.array(R, dtype=np.int)

    last = candidates[-1]
    changepoints = [last]
    while last > 0:
        last = candidates[last]
        changepoints.append(last)

    return sorted(changepoints)

