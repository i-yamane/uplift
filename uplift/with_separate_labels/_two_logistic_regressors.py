from ._utils import UpliftSepMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from ._two_classifier_util import TwoClassifiers


class TwoLogisticRegressors(UpliftSepMixin):
    def __init__(self):
        self.cv_w = 5
        self.Cs_w = [1000, 100, 10, 1]
        self.cv_z = 5
        self.Cs_z = [1000, 100, 10, 1]

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        self.model_w_ = LogisticRegressionCV(Cs=self.Cs_w, cv=self.cv_w)
        self.model_z_ = LogisticRegressionCV(Cs=self.Cs_w, cv=self.cv_w)

        self._model_ = TwoClassifiers(model_w=self.model_w_, model_z=self.model_z_)
        self._model_.fit_y_t_k(x, xy, y, ky, my, xt, t, kt, nt)

        return self

    def ranking_score(self, x):
        return (self.model_z_.predict_proba(x)[:, 1] - 0.5)\
               / (self.model_w_.predict_proba(x)[:, 1] - 0.5)

