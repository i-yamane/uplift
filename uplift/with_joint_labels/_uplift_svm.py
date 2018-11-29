import numpy as np
import sklearn.svm as svm
import itertools as it

from sklearn.base import BaseEstimator

from ._utils import calc_AUUC, unpack


class UpliftSVMThreshold(BaseEstimator):
    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def fit(self, x, yt):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        # parameters = {'kernel': ['rbf', ], 'C': [10, 20, 30], 'gamma': [0.125, 0.25, 0.5, 1, 2, ]}
        self._model = svm.SVC(kernel='linear', class_weight=self.class_weight, tol=1E-5)
        self._model.fit(x, z)
        return self

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(model=self, x=x, y=y, t=t)

    def predict(self, x_test):
        return self._model.predict(x_test)


class UpliftRankSVM(BaseEstimator):
    def __init__(self, n_ctr_samples):
        self.n_ctr_samples = n_ctr_samples

    def fit(self, x, yt):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        x_diff, z_diff = [], []
        for i1, i2 in it.combinations(range(x.shape[0]), 2):
            if z[i1] == z[i2]:  # TODO: Should we really remove samples with z[i1] == z2[i2]?
                continue
            x_diff.append(x[i1] - x[i2])
            z_diff.append(z[i1] - z[i2])
            # add reversed differences for balance between positive and negative samples
            # TODO: Is this okay?
            x_diff.append(x[i2] - x[i1])
            z_diff.append(z[i2] - z[i1])

        self._model = svm.SVC(kernel='linear')
        self._model.fit(x_diff, z_diff)

    def predict(self, x_test):
        coef = self._model.coef_
        g = np.dot(coef, x_test.T)
        g = g[0, :]

        # take top-k samples:
        i_top = np.argpartition(g, -self.n_ctr_samples)[-self.n_ctr_samples:]

        # label the top-k samples as to-be-treated (1) the rest as control (0)
        y_hat = np.zeros(x_test.shape[0])
        # y_hat[list(map(int, i_top))] = 1
        y_hat[i_top.astype(int)] = 1

        return y_hat

