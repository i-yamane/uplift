import numpy as np
import scipy as sp
import numpy.random as random
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import sklearn.svm as svm
import itertools as it
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, zero_one_loss

from sklearn.base import BaseEstimator

from ._util import UpliftWrap, calc_AUUC, unpack


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


class TwoLogistic(BaseEstimator):
    def __init__(self, n_treated=100):
        self.n_treated = n_treated

    def fit(self, x=None, yt=None):
        y, t = unpack(yt)
        # z = np.array(y == t, dtype=int)
        # z = 2 * z - 1  # Change representation from {0, 1} to {-1, +1}

        # py[0]: p(y=1|x,t=-1), py[1]: p(y=1|x,t=1)
        self.py_ = [LogisticRegression().fit(x[t == i, :], y[t == i]) for i in [0, 1]]

        return self

    def ranking_score(self, x):
        pyhat = [self.py_[0].predict_proba(x)[:, 0], self.py_[1].predict_proba(x)[:, 0]]
        uplift = (pyhat[0] - pyhat[1])
        return uplift

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(model=self, x=x, y=y, t=t)

    def predict(self, x):
        """Predicts z given x.
        """
        z_hat = np.zeros(x.shape[0], dtype=int)
        # if self.n_treated is not None and self.threshold is None:
        #     i_top = self.rank(x)[:self.n_treated]
        # elif self.threshold is not None and self.n_treated is None:
        #     i_top = self.ranking_score(x) > self.threshold
        i_top = self.rank(x)[:self.n_treated]  # debug
        z_hat[i_top] = 1

        return z_hat

    def rank(self, x):
        score = self.ranking_score(x)
        i_ranked = np.argsort(a=score, axis=0)[::-1]
        return i_ranked

class UpliftLogisticRegression(BaseEstimator):
    def __init__(self, n_treated=None, threshold=None, cv=5):
        self.n_treated = n_treated
        self.threshold = threshold
        self.cv = cv

    def fit(self, x, yt):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        self._model_ = LogisticRegressionCV(Cs=[1000, 100, 10, 1], cv=self.cv)
        self._model_.fit(x, z)
        return self

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(model=self, x=x, y=y, t=t)

    def predict_proba(self, x):
        return self._model_.predict_proba(x)

    def predict_average_uplift(self, x):
        # Definition: p(z=1|x) = (p_T(y=1|x) - p_C(y=1|x)) / 2 + 1 / 2
        pz1 = self.predict_proba(x)[:, 1]
        return 2 * pz1 - 1

    def predict(self, x):
        """Predicts z given x.
        """
        z_hat = np.zeros(x.shape[0], dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = self.ranking_score(x) > self.threshold
        z_hat[i_top] = 1

        return z_hat

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def rank(self, x):
        score = self.ranking_score(x)
        i_ranked = np.argsort(a=score, axis=0)[::-1]
        return i_ranked


class LogisticSeparate(BaseEstimator):
    def __init__(self, n_treated=None, threshold=None, cv=3):
        self.n_treated = n_treated
        self.threshold = threshold
        self.cv = cv

    def fit(self, x, yt):
        y, t = unpack(yt)
        x_trt = x[t == 1, :]
        y_trt = y[t == 1]
        x_ctl = x[t == 0, :]
        y_ctl = y[t == 0]
        self._model_ctl_ = LogisticRegressionCV(Cs=[1000, 100, 10, 1], cv=self.cv)
        self._model_ctl_.fit(x_ctl, y_ctl)
        self._model_trt_ = LogisticRegressionCV(Cs=[1000, 100, 10, 1], cv=self.cv)
        self._model_trt_.fit(x_trt, y_trt)
        self._model_ = UpliftWrap(self.ranking_score, self.n_treated)
        return self

    def predict_average_uplift(self, x):
        return self._model_trt_.predict_proba(x)[:, 1] - self._model_ctl_.predict_proba(x)[:, 1]

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict(self, x):
        return self._model_.predict(x)

    def rank(self, x):
        return self._model_.rank(x)

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(self, x, y, t)


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

