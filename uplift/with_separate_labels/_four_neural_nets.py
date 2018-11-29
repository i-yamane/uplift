import numpy as np
from sklearn.base import BaseEstimator
import chainer.links as L
import chainer.functions as F
from chainer import link
#from skchainer import ChainerRegresser

from ._utils import UpliftSepMixin
from ._neural_nets_util import Sklearnify, SklearnifyRegressor
from ._neural_nets_util import NN


class FourNNClassifiers(UpliftSepMixin):
    def __init__(self, n_hidden=10, lr=0.001, n_epochs=1000):
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_epochs = n_epochs

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        """ py[0]: p(y=1|x,k=-1), py[1]: p(y=1|x,k=1), pt[0]: p(t=1|x,k=-1), pt[1]: p(t=1|x,k=1)
        """
        y = 2 * y - 1  # {0, 1} => {-1, +1}
        t = 2 * t - 1  # {0, 1} => {-1, +1}

        self.py_ = [Sklearnify(L.Classifier(NN(n_mid_units=self.n_hidden, n_out=2)), n_epochs=self.n_epochs, lr=self.lr).fit(xy[ky == i, :], y[ky == i]) for i in [-1, 1]]
        self.pt_ = [Sklearnify(L.Classifier(NN(n_mid_units=self.n_hidden, n_out=2)), n_epochs=self.n_epochs, lr=self.lr).fit(xt[kt == i, :], t[kt == i]) for i in [-1, 1]]

    def ranking_score(self, x):
        pyhat = [self.py_[0].decision_function(x), self.py_[1].decision_function(x)]
        pthat = [self.pt_[0].decision_function(x), self.pt_[1].decision_function(x)]
        uplift = (pyhat[0] - pyhat[1]) / (pthat[0] - pthat[1])
        return uplift


class FourNNRegressors(UpliftSepMixin):
    def __init__(self):
        pass

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        """ py[0]: p(y=1|x,k=-1), py[1]: p(y=1|x,k=1), pt[0]: p(t=1|x,k=-1), pt[1]: p(t=1|x,k=1)
        """
        t = 2 * t - 1  # {0, 1} => {-1, +1}

        self.py_ = [SklearnifyRegressor(
                        Regressor(NN(n_mid_units=10, n_out=1))
                        , n_epochs=1000
                        , lr=0.001
                        , reg_level=0.0005
                    ).fit(xy[ky == i, :], y[ky == i])
                    for i in [-1, 1]]
        self.pt_ = [Sklearnify(
                        L.Classifier(NN(n_mid_units=10, n_out=2))
                        , n_epochs=1000
                        , lr=0.001
                        , reg_level=0.0005
                    ).fit(xt[kt == i, :], t[kt == i])
                    for i in [-1, 1]]

    def ranking_score(self, x):
        eyhat = [self.py_[0].predict(x), self.py_[1].predict(x)]
        pthat = [self.pt_[0].predict(x), self.pt_[1].predict(x)]
        uplift = 2 * (eyhat[0] - eyhat[1]) / (pthat[0] - pthat[1])
        return uplift


#class FourNNRegressors(UpliftSepMixin):
#    def __init__(self):
#        pass
#
#    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
#        """ py[0]: p(y=1|x,k=-1), py[1]: p(y=1|x,k=1), pt[0]: p(t=1|x,k=-1), pt[1]: p(t=1|x,k=1)
#        """
#        t = 2 * t - 1  # {0, 1} => {-1, +1}
#        self.py_ = [NNRegressors(
#                        network_params=dict(
#                            n_mid_units=10,
#                            n_out=1
#                            , n_epochs=1000
#                            , lr=0.001
#                            , reg_level=0.0005
#                        )
#                    ).fit(xy[ky == i, :], y[ky == i])
#                    for i in [-1, 1]]
#        self.pt_ = [NNRegressors(
#                        network_params=dict(
#                            n_mid_units=10, n_out=1)
#                            , n_epochs=1000
#                            , lr=0.001
#                            , reg_level=0.0005
#                        )
#                    ).fit(xt[kt == i, :], t[kt == i])
#                    for i in [-1, 1]]
#
#    def ranking_score(self, x):
#        pyhat = [self.py_[0].decision_function(x), self.py_[1].decision_function(x)]
#        pthat = [self.pt_[0].decision_function(x), self.pt_[1].decision_function(x)]
#        uplift = (pyhat[0] - pyhat[1]) / (pthat[0] - pthat[1])
#        return uplift


#class NNRegressors(ChainerRegresser):
#    def _setup_net(self, **params):
#        return NN(**params)
#
#    def _forward(self, x, train=False):
#        self.network(x)
#
#    def _loss_func(self, y, t):
#        return F.mean_squared_error(y, t)


class Regressor(L.Classifier):
    def __init__(self
            , predictor
            , lossfun=lambda y, t: F.mean_squared_error(y[:,0], t)
            , accfun=lambda y, t: F.mean_squared_error(y[:,0], t)):
            #, lossfun=F.mean_squared_error
            #, accfun=F.mean_squared_error):
        super(Regressor, self).__init__(
                predictor=predictor
                , lossfun=lossfun
                , accfun=accfun)

