import numpy as np
from ._util import unpack, calc_AUUC
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class UpliftLSCV(BaseEstimator):
    params_grid = [{'reg_level': [0.01, 0.1, 1, 10],
                    'band_width': [0.1, 0.5, 1, 5, 10, 50]}]

    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=None,
                 band_width=None,
                 cv=5):
        self.n_treated = n_treated
        self.threshold = threshold
        self.reg_level = reg_level
        self.band_width = band_width
        self.cv = cv

    def fit(self, x, yt):
        self.model_ = GridSearchCV(
            UpliftLS(
                n_treated=self.n_treated,
                threshold=self.threshold,
                reg_level=self.reg_level,
                band_width=self.band_width),
            param_grid=self.params_grid,
            cv=self.cv)
        self.model_.fit(x, yt)

    def predict(self, x):
        return self.model_.predict(x)

    def rank(self, x):
        return self.model_.best_estimator_.rank(x)


class UpliftLS(BaseEstimator):
    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=0.001,
                 band_width=10):
        self.n_treated = n_treated
        self.threshold = threshold
        self.reg_level = reg_level
        self.band_width = band_width

    def fit(self, x, yt):
        n, dim = x.shape
        self.n_basis_ = np.min((100, n))
        self.n_basis_ = dim
        self.v_ = x[:self.n_basis_, :]
        phi = self.phi(x)
        y, t = unpack(yt)
        t = 2 * t - 1  # Change coding from {0, 1} to {-1, +1}
        # y should be in {0, 1}
        y = y.reshape((n, 1))
        t = t.reshape((n, 1))
        h = np.mean(y * t * phi, axis=0)
        H = np.dot(phi.T, phi) / n
        self.w_ = np.linalg.solve(H + self.reg_level * np.eye(self.n_basis_), h)

    def phi(self, x):
        # return x
        n, dim = x.shape
        vx = np.dot(x, self.v_.T)
        xx = np.tile(np.sum(x ** 2, axis=1), (self.n_basis_, 1))
        vv = np.tile(np.sum(self.v_ ** 2, axis=1), (n, 1))
        distmat = xx.T - 2 * vx + vv
        phi = np.exp(- distmat / (2 * self.band_width))
        return phi

    def predict(self, x):
        z_hat = np.zeros((x.shape[0], 1), dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = self.ranking_score(x) > self.threshold
        z_hat[i_top] = 1

        return z_hat

    def rank(self, x):
        score = self.ranking_score(x)
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict_average_uplift(self, x):
        return np.dot(self.phi(x), self.w_)

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(self, x=x, y=y, t=t)

