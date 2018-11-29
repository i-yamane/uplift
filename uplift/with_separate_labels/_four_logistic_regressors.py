import numpy as np

from ._utils import UpliftSepMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


class FourLogistic(UpliftSepMixin):
    def __init__(self):
        pass

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        # py[0]: p(y=1|x,k=-1), py[1]: p(y=1|x,k=1), pt[0]: p(t=1|x,k=-1), pt[1]: p(t=1|x,k=1)
        # self.py_ = [LogisticRegression(fit_intercept=False).fit(xy[ky == i, :], t[ky == i]) for i in [-1, 1]]
        # self.pt_ = [LogisticRegression(fit_intercept=False).fit(xt[kt == i, :], t[kt == i]) for i in [-1, 1]]
        self.pt_ = [None] * 2
        self.py_ = [None] * 2
        for i, kk in enumerate([-1, 1]):
            tset = list(set(t[kt == kk]))
            yset = list(set(y[ky == kk]))

            #if tset != [0, 1]:
            #    import pudb; pudb.set_trace()
            #    raise ValueError(
            #            't must be in {0, 1}.'
            #            , 'set(t): {}'.format(tset))

            if len(tset) == 2:
                self.pt_[i] = LogisticRegression().fit(xt[kt == kk, :], t[kt == kk])
            else:
                tt = tset[0]
                #pt1 = (1+tt)/2
                pt1 = tt
                self.pt_[i] = ConstantPredictor(p1=pt1)
            if len(yset) == 2:
                self.py_[i] = LogisticRegression().fit(xy[ky == kk, :], y[ky == kk])
            else:
                yy = yset[0]
                #py1 = (1+yy)/2
                py1 = yy
                self.py_[i] = ConstantPredictor(p1=py1)

        # self.pt_ = [LogisticRegression().fit(xt[kt == i, :], t[kt == i]) for i in [-1, 1] if len(set(t[kt == i])) == 2 else None]
        # self.py_ = [LogisticRegression().fit(xy[ky == i, :], y[ky == i]) for i in [-1, 1] if len(set(y[ky== i])) == 2 else None]

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

