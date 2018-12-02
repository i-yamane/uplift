import numpy as np

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

import chainer
from chainer import optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from ._utils import mean_err, zero_one, calc_AUUC, unpack


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

        for epoch in range(self.n_epochs):
            self.slope += self.slope_increment
            if self.online:
                perm = range(n)
            else:
                perm = np.random.permutation(n)  # if aim is batch optimization

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
            phi = x.astype(np.float32)
            return self.l1(phi)

