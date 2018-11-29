import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator

import chainer
from chainer import optimizers, iterators, report, training
from chainer import Chain
from chainer.training import extensions
import chainer.functions as F

from ._neural_nets_util import NN

from ._utils import UpliftSepMixin
from ._utils import separate_xlsk


class USepSq(BaseEstimator, UpliftSepMixin):
    """ Elements of y, t, k must be one of {-1, +1, np.nan}
    """
    def __init__(self,
                 band_width=1):
        self.band_width = band_width

    def fit_y_t_k(self, x, xy, yy, ky, my, xt, tt, kt, nt):
        n, _ = x.shape

        t = 2 * t - 1  # {0, 1} => {-1, +1}

        self.n_basis_ = np.min((100, n))
        ids = np.random.permutation(n)
        ids = ids[:self.n_basis_]
        self.v_ = x[ids, :]

        yy = yy[:, np.newaxis]
        tt = tt[:, np.newaxis]
        kt = kt[:, np.newaxis]
        ky = ky[:, np.newaxis]
        nt = nt[:, np.newaxis]
        my = my[:, np.newaxis]

        phiy = self.phi(xy)
        phit = self.phi(xt)
        wphit = tt * kt * phit / nt

        h = np.sum(yy * ky * phiy / my, axis=0)
        G = np.dot(wphit.T, phit)

        eval, evec = np.linalg.eig(G)

        lam = 0.001
        Greg = evec.dot(np.diag(eval + lam * np.sign(eval))).dot(evec.T)

        self.w_ = np.linalg.solve(Greg, h)

        return self

    def phi(self, x):
        # return x
        n, dim = x.shape
        vx = np.dot(x, self.v_.T)
        xx = np.tile(np.sum(x ** 2, axis=1), (self.n_basis_, 1))
        vv = np.tile(np.sum(self.v_ ** 2, axis=1), (n, 1))
        distmat = xx.T - 2 * vx + vv
        phi = np.exp(- distmat / (2 * self.band_width))
        return phi

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict_average_uplift(self, x):
        return np.dot(self.phi(x), self.w_)

    def score(self, x, lsk):
        pass


class USepLSCV(BaseEstimator):
    default_params_grid = [{
        'reg_level': [1e-3],                # 'reg_level': [1e-7, 1e-6, 1e-5, 1e-3],
        'band_width': [0.000001, 1],  # 'band_width': [0.1, 0.25, 0.5, 1, 5, 10]
    }]

    def __init__(self,
                 reg_level=None,
                 band_width=None,
                 params_grid=None,
                 cv=5):
        self.reg_level = reg_level
        self.band_width = band_width
        self.cv = cv
        if params_grid is None:
            self.params_grid = USepLSCV.default_params_grid
        else:
            self.params_grid = params_grid

    def fit(self, x, lsk):
        self.model_ = GridSearchCV(
            LSGradNormLinear(
                reg_level=self.reg_level,
                band_width=self.band_width),
            param_grid=self.params_grid,
            cv=self.cv)
        self.model_.fit(x, lsk)

    def predict(self, x):
        return self.model_.predict(x)

    def rank(self, x):
        return self.model_.best_estimator_.rank(x)


class LSGradNormLinear(BaseEstimator, UpliftSepMixin):
    """ Elements of y, t, k must be in {-1, +1, np.nan}
    """
    def __init__(self,
                 reg_level=1E-2,
                 band_width=2.5,
                 n_b=1000):
        self.reg_level = reg_level
        self.band_width = band_width
        self.n_b = n_b

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        n, _ = x.shape

        t = 2 * t - 1  # {0, 1} => {-1, +1}

        self.n_basis_ = np.min((self.n_b, n))
        ids = np.random.permutation(n)
        ids = ids[:self.n_basis_]
        self.v_ = x[ids, :]

        y = y[:, np.newaxis]
        t = t[:, np.newaxis]
        kt = kt[:, np.newaxis]
        ky = ky[:, np.newaxis]
        nt = nt[:, np.newaxis]
        my = my[:, np.newaxis]

        phiy = self.phi(xy)
        phit = self.phi(xt)
        wphit = t * kt * phit / nt

        h = np.sum(y * ky * phiy / my, axis=0)
        G = np.dot(wphit.T, phit)

        self.w_ = np.linalg.solve(G.dot(G) + self.reg_level * np.eye(self.n_basis_), G.dot(h))

        return self

    def phi(self, x):
        # self.n_basis_ = 2; return x
        n, dim = x.shape
        vx = np.dot(x, self.v_.T)
        xx = np.tile(np.sum(x ** 2, axis=1), (self.n_basis_, 1))
        vv = np.tile(np.sum(self.v_ ** 2, axis=1), (n, 1))
        distmat = xx.T - 2 * vx + vv
        phi = np.exp(- distmat / (2 * self.band_width))
        return phi

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict_average_uplift(self, x):
        return np.dot(self.phi(x), self.w_)

    def score(self, x, lsk):
        xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)

        my = np.array([np.sum((ky == kk).astype(np.int)) for kk in ky], dtype=np.int32)
        nt = np.array([np.sum((kt == kk).astype(np.int)) for kk in kt], dtype=np.int32)

        y = y[:, np.newaxis]
        t = t[:, np.newaxis]
        kt = kt[:, np.newaxis]
        ky = ky[:, np.newaxis]
        nt = nt[:, np.newaxis]
        my = my[:, np.newaxis]

        phiy = self.phi(xy)
        phit = self.phi(xt)
        wphit = t * kt * phit / nt

        h = np.sum(y * ky * phiy / my, axis=0)
        G = np.dot(wphit.T, phit)

        return -np.linalg.norm(G.dot(self.w_) - h)


class USepLSModified(BaseEstimator, UpliftSepMixin):
    """ Elements of y, t, k must be one of {-1, +1, np.nan}
    """
    def __init__(self,
                 reg_level=0.0001,
                 band_width=1):
        self.reg_level = reg_level
        self.band_width = band_width

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        n, _ = x.shape

        t = 2 * t - 1  # {0, 1} => {-1, +1}

        self.n_basis_ = np.min((1000, n))
        ids = np.random.permutation(n)
        ids = ids[:self.n_basis_]
        self.v_ = x[ids, :]

        y = y[:, np.newaxis]
        t = t[:, np.newaxis]
        kt = kt[:, np.newaxis]
        ky = ky[:, np.newaxis]
        nt = nt[:, np.newaxis]
        my = my[:, np.newaxis]

        phiy = self.phi(xy)
        phit = self.phi(xt)

        h = np.sum(y * ky / my, axis=0)
        wphit = np.sum(t * kt * phit / nt, axis=0)

        # self.w_ = h / (wphit + self.reg_level)

        self.w_ = h * wphit / (wphit ** 2 + self.reg_level * np.ones(shape=(self.n_basis_,)))
        v = wphit ** 2 + self.reg_level * np.ones(shape=(self.n_basis_,))

        # self.w_ = np.linalg.solve(G.dot(G) + self.reg_level * np.eye(self.n_basis_), G.dot(h))

        return self

    def phi(self, x):
        # self.n_basis_ = 2; return x
        n, dim = x.shape
        vx = np.dot(x, self.v_.T)
        xx = np.tile(np.sum(x ** 2, axis=1), (self.n_basis_, 1))
        vv = np.tile(np.sum(self.v_ ** 2, axis=1), (n, 1))
        distmat = xx.T - 2 * vx + vv
        phi = np.exp(- distmat / (2 * self.band_width))
        return phi

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict_average_uplift(self, x):
        return np.dot(self.phi(x), self.w_)

    def score(self, x, lsk):
        xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)

        my = np.array([np.sum(np.int(ky == kk)) for kk in ky], dtype=np.int32)
        nt = np.array([np.sum(np.int(kt == kk)) for kk in kt], dtype=np.int32)

        y = y[:, np.newaxis]
        t = t[:, np.newaxis]
        kt = kt[:, np.newaxis]
        ky = ky[:, np.newaxis]
        nt = nt[:, np.newaxis]
        my = my[:, np.newaxis]

        phiy = self.phi(xy)
        phit = self.phi(xt)
        wphit = t * kt * phit / nt

        h = np.sum(y * ky * phiy / my, axis=0)
        G = np.dot(wphit.T, phit)

        return -np.linalg.norm(G.dot(self.w_) - h)


class MinGradNormNeuralNet(UpliftSepMixin):
    def __init__(self, n_epochs=1000, batch_size=128, debug=False, validation_iter=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.debug = debug
        self.validation_iter = validation_iter

    def fit(self, x, lsk):
        x = x.astype(np.float32)
        lsk = lsk.astype(np.int32)
        self.model_ = FuncJ(predictor=NN(n_mid_units=5, n_out=1))

        train = chainer.datasets.tuple_dataset.TupleDataset(x, lsk)
        train_iter = iterators.SerialIterator(train, batch_size=self.batch_size, shuffle=True)
        optimizer = optimizers.SGD(lr=0.0001)
        #optimizer = optimizers.Adam()
        # optimizer = optimizers.AdaDelta(rho=0.9999)
        # optimizer = optimizers.MomentumSGD(lr=0.0001)
        optimizer.setup(self.model_)
        # optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=100))
        optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

        # updater = training.StandardUpdater(train_iter, optimizer)
        # trainer = training.Trainer(updater, (self.n_epochs, 'epoch'), out='result')

        # if self.debug:
        #     if self.validation_iter:
        #         trainer.extend(extensions.Evaluator(self.validation_iter, self.model))
        #     trainer.extend(extensions.LogReport())
        #     trainer.extend(extensions.PrintReport(
        #         ['epoch', 'main/loss', 'validation/main/loss',
        #          'main/accuracy', 'validation/main/accuracy']))
        #     trainer.extend(extensions.ProgressBar())
        # trainer.run()

        losses = []
        while train_iter.epoch < self.n_epochs:
            train_batch = train_iter.next()
            x_bat, lsk_bat = chainer.dataset.concat_examples(train_batch)
            self.model_.cleargrads()
            loss = self.model_(x_bat, lsk_bat)
            losses.append(loss.data)
            loss.backward()
            optimizer.update()

        plt.plot(losses)

        return self

    def ranking_score(self, x):
        x = x.astype(np.float32)
        h_out = self.model_.predictor(x)
        return h_out.data


class FuncJ(Chain):
    def __init__(self, predictor):
        super(FuncJ, self).__init__()
        with self.init_scope():
            self.predictor = predictor


    def __call__(self, x, lsk):
        xz, y, ky, xw, t, kt = separate_xlsk(x, lsk)
        z = ky * y
        z = z[:, None]
        w = kt * t
        w = w[:, None]

        fz = self.predictor(xz)
        fw = self.predictor(xw)

        self.cleargrads()
        j = F.mean(w * (1 + fw * fw)) - 2 * F.mean(z * fz)
        j.backward(enable_double_backprop=True)
        del j

        grad_j_normsq = 0
        for param in self.params():
            grad_j_normsq += F.sum(param.grad_var ** 2)

        report({'loss': grad_j_normsq}, self)
        return grad_j_normsq
