import numpy as np

from ._utils import UpliftSepMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


class FourLogistic(UpliftSepMixin):
    def __init__(self):
        pass

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        self.pt_ = [None] * 2
        self.py_ = [None] * 2
        for i, kk in enumerate([-1, 1]):
            tset = list(set(t[kt == kk]))
            yset = list(set(y[ky == kk]))

            if len(tset) == 2:
                self.pt_[i] = LogisticRegression().fit(xt[kt == kk, :], t[kt == kk])
            else:
                tt = tset[0]
                pt1 = tt
                self.pt_[i] = ConstantPredictor(p1=pt1)
            if len(yset) == 2:
                self.py_[i] = LogisticRegression().fit(xy[ky == kk, :], y[ky == kk])
            else:
                yy = yset[0]
                py1 = yy
                self.py_[i] = ConstantPredictor(p1=py1)

    def ranking_score(self, x):
        pyhat = [self.py_[0].predict_proba(x)[:, 0], self.py_[1].predict_proba(x)[:, 0]]
        pthat = [self.pt_[0].predict_proba(x)[:, 0], self.pt_[1].predict_proba(x)[:, 0]]
        uplift = 2 * (pyhat[0] - pyhat[1]) / (pthat[0] - pthat[1])
        return uplift


class ConstantPredictor:
    def __init__(self, p1):
        self.p1 = p1

    def predict_proba(self, x):
        n = x.shape[0]
        return np.c_[(1-self.p1)*np.ones(n), self.p1*np.ones(n)]

