import numpy as np

from ._utils import UpliftSepMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Ridge
from ._four_logistic_regressors import ConstantPredictor


class FourRidgeRegressors(UpliftSepMixin):
    def __init__(self):
        pass

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        """ py[0]: p(y=1|x,k=-1), py[1]: p(y=1|x,k=1), pt[0]: p(t=1|x,k=-1), pt[1]: p(t=1|x,k=1)
        """

        self.pt_ = [None] * 2
        self.py_ = [None] * 2
        for i, kk in enumerate([-1, 1]):
            tset = list(set(t[kt == kk]))
            yset = list(set(y[ky == kk]))

            if len(tset) == 2:
                self.pt_[i] = LogisticRegression().fit(xt[kt == kk, :], t[kt == kk])
            else:
                tt = tset[0]
                #pt1 = (1+tt)/2
                pt1 = tt
                self.pt_[i] = ConstantPredictor(p1=pt1)
            if len(yset) == 2:
                self.py_[i] = Ridge(alpha=0.1).fit(xy[ky == i, :], y[ky == i])
            else:
                yy = yset[0]
                self.py_[i] = ConstantRegressor(c=yy)

    def ranking_score(self, x):
        pyhat = [self.py_[0].predict(x), self.py_[1].predict(x)]
        pthat = [self.pt_[0].predict_proba(x)[:, 0], self.pt_[1].predict_proba(x)[:, 0]]
        uplift = 2 * (pyhat[0] - pyhat[1]) / (pthat[0] - pthat[1])
        return uplift


class ConstantRegressor:
    def __init__(self, c):
        self.c = c

    def predict(self, x):
        n = x.shape[0]
        return self.c*np.ones(n)

