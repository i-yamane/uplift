from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from ._utils import UpliftSepMixin
from ._two_classifier_util import TwoClassifiers


class TwoSVMs(UpliftSepMixin):
    def __init__(self):
        self.cv = 5
        self.params_grid = {'kernel': ['rbf', ], 'C': [10, 25, 50, 100], 'gamma': [0.125, 0.25, 0.5, 1, 2]}
        self.param_distributions = {'kernel': ['rbf', ],
                                    'C': [1, 5, 10, 20, 40, 50, 75, 100],
                                    'gamma': [0.05, 0.125, 0.25, 0.5, 1, 2, 5, 10, 50, 100]}

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        # self.model_w_ = GridSearchCV(svm.SVC(), param_grid=self.params_grid, cv=self.cv)
        # self.model_z_ = GridSearchCV(svm.SVC(), param_grid=self.params_grid, cv=self.cv)
        self.model_w_ = RandomizedSearchCV(SVC(), param_distributions=self.param_distributions, cv=self.cv, n_iter=20)
        self.model_z_ = RandomizedSearchCV(SVC(), param_distributions=self.param_distributions, cv=self.cv, n_iter=20)

        self._model_ = TwoClassifiers(model_w=self.model_w_, model_z=self.model_z_)
        self._model_.fit_y_t_k(x, xy, y, ky, my, xt, t, kt, nt)

        return self

    def ranking_score(self, x):
        return self._model_.ranking_score(x)  # The output may not be meaningful
