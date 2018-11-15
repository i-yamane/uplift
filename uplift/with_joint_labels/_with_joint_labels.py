import numpy as np
import scipy as sp
import numpy.random as random
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import sklearn.svm as svm
import itertools as it
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, zero_one_loss

from sklearn.base import BaseEstimator


import chainer
from chainer import serializers, cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import sympy as smp
# from sympy import init_printing;
# init_printing()

import matplotlib.pyplot as plt

import code

from ._util import mean_err, zero_one, sym2numpy_func, check_grad, taylor_err, UpliftMixin, UpliftWrap, calc_AUUC, max_uplift, calc_AUUC2, max_uplift2, calc_actual_uplift, plot_uplift_curve, plot_diff_prob, unpack, calc_uplift2, calc_uplift


class UpliftRampThresholdSGDManyTimes(BaseEstimator):
    def __init__(self, class_weight=None,
                 online=False,
                 n_epochs=1000,
                 batch_size=50,
                 n_inits=10,
                 use_hard_sigmoid=False):
        self.class_weight = class_weight
        self.online = online
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._model_best = None
        self._err_best = None
        self.n_inits = n_inits
        self.use_hard_sigmoid = use_hard_sigmoid

    def fit(self, x, yt):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        for i in range(self.n_inits):
            model_tmp = UpliftRampThresholdSGD(class_weight=self.class_weight,
                                               online=self.online,
                                               n_epochs=self.n_epochs,
                                               batch_size=self.batch_size,
                                               use_hard_sigmoid=self.use_hard_sigmoid)
            model_tmp.fit(x, z)
            z_hat = model_tmp.predict(x)
            err_tmp = mean_err(z_true=z, z_hat=z_hat)
            if self._err_best is None or err_tmp < self._err_best:
                self._model_best = model_tmp
                self._err_best = err_tmp

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(model=self, x=x, y=y, t=t)

    def predict(self, x_test):
        """ Predicts the z labels for the input x_test
        """
        z_hat = self._model_best.predict(x_test)
        return z_hat


class UpliftSigCV(BaseEstimator):
    params_grid = [{'reg_level': [1e-8, 1e-4, 1e-2, 1e-1]}]

    def __init__(self,
                 class_weight=None,
                 online=False,
                 n_epochs=10,
                 batch_size=10,
                 reg_level=0.0001,
                 slope=1,
                 use_hard_sigmoid=False,
                 rho=0.999999,
                 lr=0.1,
                 slope_increment=False,
                 logistic=False,
                 init_with=None):
        self.class_weight = class_weight
        self.online = online
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_level = reg_level
        self.slope = slope
        self.use_hard_sigmoid = use_hard_sigmoid
        self.rho = rho
        self.lr = lr
        self.slope_increment = slope_increment
        self.logistic = logistic
        self.init_with = init_with

    def fit(self, x, yt):
        self.model_ = GridSearchCV(UpliftRampThresholdSGD(**self.__dict__),
                                   param_grid=self.params_grid,
                                   cv=self.cv)
        self.model_.fit(x, yt)
        return self

    def predict(self, x):
        return self.model_.predict(x)

    def rank(self, x):
        return self.model_.rank(x)


class UpliftRampThresholdSGD(BaseEstimator):
    def __init__(self,
                 class_weight=None,
                 online=False,
                 n_epochs=10,
                 batch_size=10,
                 reg_level=0.0001,
                 slope=1,
                 use_hard_sigmoid=False,
                 rho=0.999999,
                 lr=0.1,
                 slope_increment=False,
                 logistic=False,
                 init_with=None):
        self.class_weight = class_weight
        self.online = online
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_level = reg_level
        self.slope = slope
        self.use_hard_sigmoid = use_hard_sigmoid
        self.rho = rho
        self.lr = lr
        self.slope_increment = slope_increment
        self.logistic = logistic
        self.init_with = init_with

    def fit(self, x, yt):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        z = 2 * z - 1  # Change representation from {0, 1} to {-1, +1}
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        n = len(z)
        if self.init_with == 'logistic':
            prototype = LogisticRegression()
            prototype.fit(x, z)
            self._model = UpliftRampThresholdSGD.PredictiveFunc(dim_out=1, init_l1=L.Linear(
                in_size=None,
                out_size=1,
                initialW=prototype.coef_.astype(np.float32),
                initial_bias=prototype.intercept_.astype(np.float32)))
        else:
            self._model = UpliftRampThresholdSGD.PredictiveFunc(dim_out=1)
        # self._model = UpliftRampThresholdSGD.Net33(n_2ndunits=3, dim_out=1)
        # opt = optimizers.SGD(lr=self.lr)
        # opt = optimizers.AdaDelta(rho=self.rho)
        # opt = optimizers.RMSprop()
        opt = optimizers.Adam()
        # opt.use_cleargrads()  # Deprecated
        opt.setup(self._model)
        opt.add_hook(chainer.optimizer.WeightDecay(self.reg_level))

        # chainer does not support float64 but float32
        x32 = x.astype(np.float32)
        z32 = z.reshape(n, 1)
        del x, z

        # set the weight values
        if self.class_weight is None:
            r32 = 0.5 * np.ones(z32.shape)
        else:
            f = (z32 > 0).astype(np.int)
            f2r = np.array([self.class_weight[0], self.class_weight[1]])
            r32 = f2r[f]
        r32 = r32.astype(np.float32).reshape(n, 1)

        Debug.print(np.sum((z32 == np.sign(self._model(x32).data)) * 1.0 / len(z32)), message='before')

        for epoch in range(self.n_epochs):
            self.slope += self.slope_increment
            if self.online:
                perm = range(n)
            else:
                perm = np.random.permutation(n)  # if aim is batch optimization

            sum_loss = 0  # for debugging
            for i in range(0, n, self.batch_size):
                # Reguire: 0 <= i < n_train and (i - 0) % batch_size == 0.
                x_batch = x32[perm[i: i + self.batch_size], :]
                z_batch = z32[perm[i: i + self.batch_size]]
                r_batch = r32[perm[i: i + self.batch_size]]

                if self.use_hard_sigmoid:
                    loss = F.sum(r_batch * F.hard_sigmoid(- self.slope * z_batch * self._model(x_batch))) / len(z_batch) / self.slope
                else:
                    if self.logistic:
                        loss = F.sum(F.log(1 + F.exp(- z_batch * self._model(x_batch)))) / len(z_batch)
                    else:
                        # loss = F.sum(r_batch * F.sigmoid(- self.slope * z_batch * self._model(x_batch))) / len(z_batch) / self.slope
                        # loss = - F.sum(r_batch * self.slope * z_batch * self._model(x_batch)) / len(z_batch) / self.slope
                        loss = F.sum(1 / (1 + F.exp(self.slope * z_batch * self._model(x_batch)))) / (len(z_batch) * self.slope)
                self._model.cleargrads()
                loss.backward()
                opt.update()

                sum_loss += loss.data  # for debugging

            if False and epoch % 2 == 0:
                print((r_batch * F.sigmoid(- self.slope * z_batch * self._model(x_batch))).data.T)
                print(sum_loss / self.batch_size)  # for debugging
                #
                # print(self._model.l1.W.data) # for debugging
                # print(self._model.l1.b.data) # for debugging

        Debug.print(np.sum((z32 == np.sign(self._model(x32).data)) * 1.0 / len(z32)), message='after')
        Debug.print(self._model.l1.W.data, message='W learned')
        Debug.print(self._model.l1.b.data, message='b learned')
        # print(np.array([z == zero_one(zz) for z, zz in zip(z32, self._model(x32))]))
        # code.interact(local=dict(globals(), **locals()))

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(self, x=x, y=y, t=t)

    def predict(self, x):
        """ Predicts the class labels for the input x_test
        """
        z_hat = self._model(x)
        z_hat = np.array([zero_one(zz) for zz in z_hat])
        return z_hat

    def rank(self, x):
        score = self.ranking_score(x)
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)

    def ranking_score(self, x):
        return self._model(x).data

    class Net33(Chain):
        # initializer = chainer.initializers.HeNormal()
        def __init__(self, n_2ndunits=3, dim_out=1):
            super(UpliftRampThresholdSGD.Net33, self).__init__(
                l1=L.Linear(None, n_2ndunits),
                l2=L.Linear(n_2ndunits, dim_out)
            )

        def __call__(self, x):
            phi = x.astype(np.float32)
            h1 = F.relu(self.l1(phi))
            h2 = self.l2(h1)
            return h2

    class PredictiveFunc(Chain):
        def __init__(self, dim_out, init_l1=None):
            # Linear-in-parameter model
            if init_l1 is None:
                init_l1 = L.Linear(None, dim_out)
            super(UpliftRampThresholdSGD.PredictiveFunc, self).__init__(
                l1=init_l1
            )

        def __call__(self, x):
            # chainer does not support float64 but float32
            phi = x.astype(np.float32)  # TODO: Feature mapping here
            return self.l1(phi)


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

    def ranking_score(self, x):
        raise ValueError()


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

