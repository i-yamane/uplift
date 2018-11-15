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


class UpliftCombMLECV(BaseEstimator):
    params_grid = [{'reg_level': [0.001, 0.01, 0.1, 1]}]

    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=0.001,
                 n_epochs=300,
                 batch_size=10,
                 cv=5):
        self.n_treated = n_treated
        self.threshold = threshold
        self.reg_level = reg_level
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cv = cv

    def fit(self, x, yt):
        self.model_ = GridSearchCV(
            UpliftCombMLE(
                n_treated=self.n_treated,
                threshold=self.threshold,
                reg_level=self.reg_level,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size),
            param_grid=self.params_grid,
            cv=self.cv)
        self.model_.fit(x, yt)

    def predict(self, x):
        return self.model_.predict(x)

    def rank(self, x):
        return self.model_.best_estimator_.rank(x)


class UpliftCombMLE(BaseEstimator):
    """Uses Chainer 2.0.0
    """
    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=0.001,
                 n_epochs=300,
                 batch_size=20):
        self.n_treated = n_treated
        self.threshold = threshold
        self.reg_level = reg_level
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, x, yt):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        n, dim = x.shape

        x = x.astype(np.float32)
        z = 2 * z.reshape(n, 1).astype(int) - 1
        self.model_probz_ = UpliftCombMLE.ProbZModel(n_in=dim)

        # opt = optimizers.Adam(alpha=0.00001)
        # opt = optimizers.AdaDelta(rho=0.9, eps=0.0001)
        opt = optimizers.AdaGrad(lr=0.001)
        opt.setup(self.model_probz_)

        for epoch in range(1, self.n_epochs + 1):
            perm = np.random.permutation(n)

            sum_loss = 0
            for i in range(0, n, self.batch_size):
                x_batch = x[perm[i: i + self.batch_size]]
                z_batch = z[perm[i: i + self.batch_size]]

                loss = - F.sum(F.log(self.model_probz_(x_batch, z_batch))) \
                       + self.reg_level * F.sum(self.model_probz_.f_ctl.W ** 2)\
                       + self.reg_level * F.sum(self.model_probz_.f_trt.W ** 2)

                Debug.print(loss.data, message='loss')

                self.model_probz_.cleargrads()
                loss.backward()
                opt.update()

                sum_loss += loss.data
            Debug.print_var(sum_loss)

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(self, x=x, y=y, t=t)

    def predict(self, x):
        """Predicts z given x.
        """
        z_hat = np.zeros((x.shape[0], 1), dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = self.ranking_score(x) > self.threshold
        z_hat[i_top] = 1

        return z_hat

    def predict_proba(self, x):
        x = x.astype(np.float32)
        n, dim = x.shape
        z0 = np.zeros(n, dtype=np.float32)
        pz0 = self.model_probz_(x, z0).data
        pz1 = 1 - pz0
        return np.c_[pz0, pz1]

    def predict_average_uplift(self, x):
        x = x.astype(np.float32)
        n, dim = x.shape
        z1 = np.ones((n, 1), dtype=np.float32)
        pz1 = self.model_probz_(x, z1).data
        return 2 * pz1 - 1

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def rank(self, x):
        score = self.ranking_score(x)
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)

    class ProbZModel(Chain):
        def __init__(self, n_in):
            super(UpliftCombMLE.ProbZModel, self).__init__()
            with self.init_scope():
                # These are callable objects have bias terms:
                self.f_ctl = L.Linear(n_in, 1)
                self.f_trt = L.Linear(n_in, 1)

        def __call__(self, x, z):
            # Chainer does not support float64 but float32
            phi = x.astype(np.float32)  # TODO: Feature mapping here
            z = z.astype(np.float32)
            pz0_ctl = 1 / (1 + F.exp(z * self.f_ctl(phi)))
            pz1_trt = 1 / (1 + F.exp(-z * self.f_trt(phi)))
            return pz0_ctl / 2 + pz1_trt / 2


class UpliftCombinedLogisticIntercept(BaseEstimator):
    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=0.001,
                 learning_rate=10,
                 max_iter=1000,
                 tol=1e-5,
                 a_ctl_init=None,
                 a_trt_init=None,
                 debug=False):
        self.n_treated = n_treated
        self.threshold = threshold
        self.a_ctl_init = a_ctl_init
        self.a_trt_init = a_trt_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.reg_level = reg_level
        self.debug = debug

    def fit(self, x, yt):
        """Fit the model to given data.
        :param x: Input data.
        :param yt: Array that can be expressed by np.r_[y, t], where y is a vector of outputs
                    and t is a vector of treatment assignments.
        :return: The fitted model.
        """
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        n, dim = x.shape
        z = 2 * z.reshape(n, 1) - 1  # Change representation from {0, 1} to {-1, +1}
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        n, dim = x.shape
        # z = 2 * z.reshape(n, 1) - 1  # Change representation from {0, 1} to {-1, +1}
        if self.a_ctl_init is None:
            self.a_ctl_init = np.zeros((1, dim + 1))
        if self.a_trt_init is None:
            self.a_trt_init = np.zeros((1, dim + 1))
        self.a_ctl_ = self.a_ctl_init.reshape((1, dim + 1))
        self.a_trt_ = self.a_trt_init.reshape((1, dim + 1))

        loss = np.empty(self.max_iter)
        decay = np.empty(self.max_iter)
        grad = np.empty(self.max_iter)
        loop = 0
        for loop in range(self.max_iter):
            if self.debug:
                loss[loop] = self._objective(x, z, a_ctl=self.a_ctl_, a_trt=self.a_trt_)
                decay[loop] = self.learning_rate / np.sqrt(loop + 1)
                grad[loop] = np.linalg.norm(self._grad(x, z, a_ctl=self.a_ctl_, a_trt=self.a_trt_))

            param_old = np.c_[self.a_ctl_, self.a_trt_]
            a_ctl, a_trt = self._updated_param(x, z, loop + 1, a_ctl=self.a_ctl_, a_trt=self.a_trt_)
            param_new = np.c_[a_ctl, a_trt]

            self.a_ctl_, self.a_trt_ = a_ctl, a_trt

            if np.linalg.norm(param_new - param_old) < self.tol:
                break
        Debug.print_var(loop, message='Converged at ', end='\n')

        if self.debug:
            plt.subplot(211)
            plt.plot(range(loop), loss[:loop])
            plt.title('Objective')
            plt.subplot(212)
            plt.plot(range(loop), grad[:loop])
            plt.title('Gradient')
            plt.show()
            # import pdb; pdb.set_trace()

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(self, x=x, y=y, t=t)

    def _updated_param(self, x, z, k, a_ctl, a_trt):
        g_ctl, g_trt = self._grad(x, z, a_ctl=a_ctl, a_trt=a_trt)
        # a_ctl_new = a_ctl - self.learning_rate * g_ctl / np.sqrt(k)
        # a_trt_new = a_trt - self.learning_rate * g_trt / np.sqrt(k)
        # TODO
        a_ctl_new = a_ctl - self.learning_rate * g_ctl / np.sqrt(k)
        a_trt_new = a_trt - self.learning_rate * g_trt / np.sqrt(k)

        return a_ctl_new, a_trt_new

    def _grad_hardcoded(self, x, z, a_ctl, a_trt):
        v = self._calc_vals(x, a_ctl=a_ctl, a_trt=a_trt)
        f, g = v['f'], v['g']
        vF, vG, vFG = v['F'], v['G'], v['FG']
        vA, vB, vR = v['A'], v['B'], v['R']

        # Do not regularize the intercept dimension
        a_ctl_reg = a_ctl
        a_ctl_reg[:, 0] = 0
        a_trt_reg = a_trt
        a_trt_reg[:, 0] = 0

        n, dim = x.shape
        xx = np.c_[np.ones((n, 1)), x]

        # g_ctl = (2 * np.sum(xx*z*(vB*vG + vF*vFG - vG*vFG) / ((np.exp(z*np.log(vR))+1)*vA*vB), axis=0) / n) \
        #     + 2 * self.reg_level * a_ctl_reg
        # g_trt = - (2 * np.sum(xx*z*(vA*vF + vG*vFG - vF*vFG) / ((np.exp(z*np.log(vR))+1)*vA*vB), axis=0) / n) \
        #     + 2 * self.reg_level * a_trt_reg
        g_ctl = (2 * np.sum(xx*z*(vB*vG + vF*vFG - vG*vFG) / ((vR**z+1)*vA*vB), axis=0) / n) \
            + 2 * self.reg_level * a_ctl_reg
        g_trt = - (2 * np.sum(xx*z*(vA*vF + vG*vFG - vF*vFG) / ((vR**z+1)*vA*vB), axis=0) / n) \
            + 2 * self.reg_level * a_trt_reg

        return np.r_[g_ctl, g_trt]

    def _grad(self, x, z, a_ctl, a_trt, method='hardcoded'):
        if method is 'auto':
            raise NotImplementedError()
            # grad_helper = self._create_grad_auto()  # This returns a function object
        elif method is 'hardcoded':
            grad_helper = self._grad_hardcoded
        else:
            raise ValueError('Choose auto or hardcoded.')

        if self.debug:
            def func(param):
                a_ctl_, a_trt_ = param
                return self._objective(x, z, a_ctl=a_ctl_[np.newaxis, :], a_trt=a_trt_[np.newaxis, :])

            def grad(param):
                a_ctl_, a_trt_ = param
                return grad_helper(x, z, a_ctl=a_ctl_[np.newaxis, :], a_trt=a_trt_[np.newaxis, :])

            v0 = np.r_[a_ctl, a_trt]

        return grad_helper(x, z, a_ctl=a_ctl, a_trt=a_trt)

    def _objective(self, x, z, a_ctl, a_trt):
        n, _ = x.shape
        v = self._calc_vals(x, a_ctl=a_ctl, a_trt=a_trt)
        # obj = np.sum(1 + np.exp(z*np.log(v['R'])), axis=0) / n
        obj = np.sum(np.log(1 + np.exp(-z * np.log(v['R'])))) / n \
              + self.reg_level * (np.sum(a_trt ** 2) + np.sum(a_ctl ** 2))
        # obj = np.sum(np.log(1 + v['R']**(-z)), axis=0) / n\
        #       + self._regularization * (np.sum(a ** 2) + np.sum(b ** 2))
        # TODO: consider bias term for the regularization
        return obj

    def _calc_vals(self, x, a_ctl, a_trt):
        vals = {'f': self.f(x, a_trt), 'g': self.g(x, a_ctl)}
        vals['G'] = np.exp(vals['g'])
        vals['F'] = np.exp(vals['f'])
        vals['FG'] = vals['F']*vals['G']
        vals['A'] = vals['FG'] + 2*vals['G'] + 1
        vals['B'] = vals['FG'] + 2*vals['F'] + 1
        vals['R'] = vals['B'] / vals['A']
        return vals

    def predict_proba(self, x):
        v = self._calc_vals(x, a_ctl=self.a_ctl_, a_trt=self.a_trt_)

        pz0 = 1 / (1 + np.exp(v['R']))
        pz1 = 1 - pz0

        return np.c_[pz0, pz1]

    def predict_average_uplift(self, x):
        f = self.f(x, self.a_trt_)
        g = self.g(x, self.a_ctl_)
        pty1 = 1 / (1 + np.exp(-f))
        pcy1 = 1 / (1 + np.exp(-g))
        return pty1 - pcy1

    def f(self, x, a_trt):
        n, dim = x.shape
        xx = np.c_[np.ones((n, 1)), x]
        return xx.dot(self.a_trt_.T)

    def g(self, x, a_ctl):
        n, dim = x.shape
        xx = np.c_[np.ones((n, 1)), x]
        return xx.dot(self.a_ctl_.T)

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict(self, x):
        """Predicts z given x.
        """
        z_hat = np.zeros(x.shape[0], dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = self.ranking_score() > self.threshold
        else:
            raise ValueError()

        z_hat[i_top] = 1

        return z_hat

    def rank(self, x):
        score = self.ranking_score(x)
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)


class UpliftCombinedLogisticCV(BaseEstimator):
    params_grid = [{'reg_level': [0.001, 0.01, 0.1, 1]}]

    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=None,
                 learning_rate=10,
                 max_iter=1000,
                 tol=1e-5,
                 a_ctl_init=None,
                 a_trt_init=None,
                 intercept=None,
                 cv=5):
        self.n_treated = n_treated
        self.threshold = threshold
        self.reg_level = reg_level
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.a_ctl_init = a_ctl_init
        self.a_trt_init = a_trt_init
        self.intercept = intercept
        self.cv = cv

    def fit(self, x, yt):
        if self.intercept is None:
            self.method_core_ = UpliftCombinedLogistic(n_treated=self.n_treated,
                                                 threshold=self.threshold,
                                                 reg_level=self.reg_level,
                                                 learning_rate=self.learning_rate,
                                                 max_iter=self.max_iter,
                                                 tol=self.tol,
                                                 a_ctl_init=self.a_ctl_init,
                                                 a_trt_init=self.a_trt_init)
        else:
            self.method_core_ = UpliftCombinedLogisticIntercept(n_treated=self.n_treated,
                                                          threshold=self.threshold,
                                                          reg_level=self.reg_level,
                                                          learning_rate=self.learning_rate,
                                                          max_iter=self.max_iter,
                                                          tol=self.tol,
                                                          a_ctl_init=self.a_ctl_init,
                                                          a_trt_init=self.a_trt_init)
        self.model_ = GridSearchCV(self.method_core_, param_grid=self.params_grid, cv=self.cv)
        self.model_.fit(x, yt)

    def predict(self, x):
        return self.model_.predict(x)

    def rank(self, x):
        return self.model_.best_estimator_.rank(x)


class UpliftCombinedLogistic(BaseEstimator):
    def __init__(self,
                 n_treated=None,
                 threshold=None,
                 reg_level=0.001,
                 learning_rate=0.1,
                 max_iter=10000,
                 tol=1e-5,
                 a_ctl_init=None,
                 a_trt_init=None,
                 debug=False):
        self.n_treated = n_treated
        self.threshold = threshold
        self.a_ctl_init = a_ctl_init
        self.a_trt_init = a_trt_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.reg_level = reg_level
        self.debug = debug

    def fit(self, x, yt):
        """Fit the model to given data.
        :param x: Input data.
        :param yt: Array that can be expressed by np.r_[y, t], where y is a vector of outputs
                    and t is a vector of treatment assignments.
        :return: The fitted model.
        """
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        self.fit_z(x, z)
        Debug.print(self.a_ctl_)
        Debug.print(self.a_trt_)
        return self

    def fit_z(self, x, z):
        n, dim = x.shape
        z = 2 * z.reshape(n, 1) - 1  # Change representation from {0, 1} to {-1, +1}
        if self.a_ctl_init is None:
            self.a_ctl_init = np.zeros((1, dim))
        if self.a_trt_init is None:
            self.a_trt_init = np.zeros((1, dim))
        self.a_ctl_ = self.a_ctl_init.reshape((1, dim))
        self.a_trt_ = self.a_trt_init.reshape((1, dim))

        loss = np.empty(self.max_iter)
        decay = np.empty(self.max_iter)
        grad = np.empty(self.max_iter)
        loop = 0
        for loop in range(self.max_iter):
            if self.debug:
                loss[loop] = self._objective(x, z, a_ctl=self.a_ctl_, a_trt=self.a_trt_)
                decay[loop] = self.learning_rate / np.sqrt(loop + 1)
                grad[loop] = np.linalg.norm(self._grad(x, z, a_ctl=self.a_ctl_, a_trt=self.a_trt_))

            param_old = np.c_[self.a_ctl_, self.a_trt_]
            a_ctl, a_trt = self._updated_param(x, z, loop + 1, a_ctl=self.a_ctl_, a_trt=self.a_trt_)
            param_new = np.c_[a_ctl, a_trt]

            self.a_ctl_, self.a_trt_ = a_ctl, a_trt

            if np.linalg.norm(param_new - param_old) < self.tol:
                break
        Debug.print_var(loop, message='Converged at ', end='\n')

        if self.debug:
            plt.subplot(211)
            plt.plot(range(loop), loss[:loop])
            plt.title('Objective')
            plt.subplot(212)
            plt.plot(range(loop), grad[:loop])
            plt.title('Gradient')
            plt.show()
            # import pdb; pdb.set_trace()

    def score(self, x, yt):
        y, t = unpack(yt)
        return calc_AUUC(self, x=x, y=y, t=t)

    def _updated_param(self, x, z, k, a_ctl, a_trt):
        g_ctl, g_trt = self._grad(x, z, a_ctl=a_ctl, a_trt=a_trt)
        a_ctl_new = a_ctl - self.learning_rate * g_ctl / np.sqrt(np.sqrt(k))
        a_trt_new = a_trt - self.learning_rate * g_trt / np.sqrt(np.sqrt(k))

        return a_ctl_new, a_trt_new

    def _grad_hardcoded(self, x, z, a_ctl, a_trt):
        n, dim = x.shape

        v = self._calc_vals(x, a_ctl=a_ctl, a_trt=a_trt)
        f, g = v['f'], v['g']
        vF, vG, vFG = v['F'], v['G'], v['FG']
        vA, vB, vR = v['A'], v['B'], v['R']

        g_ctl = (2 * np.sum(x*z*(vB*vG + vF*vFG - vG*vFG) / ((np.exp(z*np.log(vR))+1)*vA*vB), axis=0) / n) \
            + 2 * self.reg_level * a_ctl
        g_trt = - (2 * np.sum(x*z*(vA*vF + vG*vFG - vF*vFG) / ((np.exp(z*np.log(vR))+1)*vA*vB), axis=0) / n) \
            + 2 * self.reg_level * a_trt

        return np.r_[g_ctl, g_trt]

    def _create_grad_auto(self):  # TODO: This is slow
        x, y, z, w, f, g, meta = smp.symbols('x y z w f g meta')
        p = smp.exp(y * meta) / (1 + smp.exp(y * meta))  # General form of the logistic model
        # The model for the control distribution
        pc = p.subs(meta, g)
        pcy0 = pc.subs(y, -1)
        pcy1 = pc.subs(y, 1)

        # The model for the treatment distribution
        pt = p.subs(meta, f)
        pty0 = pt.subs(y, -1)
        pty1 = pt.subs(y, 1)

        # The combined logistic model for uplift modeling
        pz0 = smp.together(pcy1 / 2 + pty0 / 2)
        pz1 = smp.together(pcy0 / 2 + pty1 / 2)

        w = smp.log(pz1 / pz0)
        obj = smp.log(1 + smp.exp(-z * w))
        dl_f = smp.simplify(smp.diff(obj, f))
        dl_g = smp.simplify(smp.diff(obj, g))

        dl_f_np = sym2numpy_func((f, g, z), dl_f, nin=3, nout=1)
        dl_g_np = sym2numpy_func((f, g, z), dl_g, nin=3, nout=1)

        def grad_auto(x_val, z_val, a_ctl, a_trt):
            n, dim = x_val.shape
            f_val = x_val.dot(a_trt.T)
            g_val = x_val.dot(a_ctl.T)
            grad_f = dl_f_np(f_val, g_val, z_val).astype('float64')
            grad_g = dl_g_np(f_val, g_val, z_val).astype('float64')
            grad_trt = grad_f * x_val
            grad_ctl = grad_g * x_val
            ave_grad_ctl = np.sum(grad_ctl, axis=0) / n
            ave_grad_trt = np.sum(grad_trt, axis=0) / n
            ave_grad_ctl = ave_grad_ctl[np.newaxis, :]
            ave_grad_trt = ave_grad_trt[np.newaxis, :]
            return np.r_[ave_grad_ctl, ave_grad_trt]

        return grad_auto

    def _grad(self, x, z, a_ctl, a_trt, method='hardcoded'):
        if method is 'auto':  # TODO: This is slow
            grad_helper = self._create_grad_auto()  # This returns a function object
        elif method is 'hardcoded':
            grad_helper = self._grad_hardcoded

        if self.debug:
            def func(param):
                a_ctl_, a_trt_ = param
                return self._objective(x, z, a_ctl=a_ctl_[np.newaxis, :], a_trt=a_trt_[np.newaxis, :])

            def grad(param):
                a_ctl_, a_trt_ = param
                return grad_helper(x, z, a_ctl=a_ctl_[np.newaxis, :], a_trt=a_trt_[np.newaxis, :])

            v0 = np.r_[a_ctl, a_trt]
            Debug.print(np.linalg.norm(grad(v0)), message='|g0|', end=', ')
            # Debug.print(np.linalg.norm(sp.optimize.check_grad(func, grad, v0)), message='(SciPy) |g_ana-g_num|.')
            Debug.print(np.linalg.norm(check_grad(func, grad, v0)), message='(Debug.cg) |g_ana-g_num|.')
            Debug.print(np.linalg.norm(taylor_err(func, grad, v0)), message='(Debug) Taylor Approx. Err.')

        return grad_helper(x, z, a_ctl=a_ctl, a_trt=a_trt)

    def _objective(self, x, z, a_ctl, a_trt):
        n, dim = x.shape
        v = self._calc_vals(x, a_ctl=a_ctl, a_trt=a_trt)
        # obj = np.sum(1 + np.exp(z*np.log(v['R'])), axis=0) / n
        obj = np.sum(np.log(1 + np.exp(-z * np.log(v['R'])))) / n \
              + self.reg_level * (np.sum(a_trt ** 2) + np.sum(a_ctl ** 2))
        # obj = np.sum(np.log(1 + v['R']**(-z)), axis=0) / n\
        #       + self._regularization * (np.sum(a ** 2) + np.sum(b ** 2))
        # TODO: consider bias term for the regularization
        return obj

    def _calc_vals(self, x, a_ctl, a_trt):
        vals = {'f': x.dot(a_trt.T), 'g': x.dot(a_ctl.T)}
        vals['G'] = np.exp(vals['g'])
        vals['F'] = np.exp(vals['f'])
        vals['FG'] = vals['F']*vals['G']
        vals['A'] = vals['FG'] + 2*vals['G'] + 1
        vals['B'] = vals['FG'] + 2*vals['F'] + 1
        vals['R'] = vals['B'] / vals['A']
        return vals

    def predict_proba(self, x):
        v = self._calc_vals(x, a_ctl=self.a_ctl_, a_trt=self.a_trt_)

        pz0 = 1 / (1 + np.exp(v['R']))
        pz1 = 1 - pz0

        return np.c_[pz0, pz1]

    def predict_average_uplift(self, x):
        f = x.dot(self.a_trt_.T)
        g = x.dot(self.a_ctl_.T)
        pty1 = 1 / (1 + np.exp(-f))
        pcy1 = 1 / (1 + np.exp(-g))
        return pty1 - pcy1

    def ranking_score(self, x):
        return self.predict_average_uplift(x)

    def predict(self, x):
        """Predicts z given x.
        """
        z_hat = np.zeros(x.shape[0], dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = self.ranking_score() > self.threshold
        else:
            raise ValueError()

        z_hat[i_top] = 1

        return z_hat

    def rank(self, x):
        score = self.ranking_score(x)
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)


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

