import numpy as np
from scipy.stats import uniform

class loguniform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def rvs(self, size=None, random_state=None):
        start = np.log(self.lower)
        stop = np.log(self.upper)
        unif = uniform(loc=start, scale=stop - start)
        return np.exp(unif.rvs(size=size, random_state=random_state))

