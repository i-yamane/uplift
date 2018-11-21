import numpy as np
import numpy.random as random
from sklearn.base import BaseEstimator

from ._util import unpack, calc_AUUC


class UpliftRandom(BaseEstimator):
    def __init__(self, n_ctr):
        self.n_ctr_samples = n_ctr

    def fit(self, x=None, yt=None):
        pass

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(model=self, x=x, y=y, t=t)

    def rank(self, x):
        i_rand = random.permutation(x.shape[0])
        return i_rand

    def predict(self, x):
        i_rand = self.rank(x)
        y_hat = np.zeros(x.shape[0])
        y_hat[i_rand[::2]] = 1

        return y_hat

