import numpy as np
import scipy as sp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator

from ._utils import UpliftSepMixin
from ._utils import separate_xlsk, separate_xlskr

import warnings
warnings.simplefilter('default')

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('Failed to import cvxpy. Do not enable `binary` option in MinMaxLinear.', ImportWarning)


class MinMaxLinear(UpliftSepMixin, BaseEstimator):
    def __init__(
            self,
            reg_level_f=1E-2,
            reg_level_g=1E-2,
            band_width_f=2.5,
            band_width_g=2.5,
            b_f=1000,
            b_g=1000,
            bfunc='gauss',
            binary=False,
            use_r=False,
            idxs_gauss=None,
            idxs_delta=None,
    ):
        self.reg_level_f = reg_level_f
        self.reg_level_g = reg_level_g
        self.band_width_f = band_width_f
        self.band_width_g = band_width_g
        self.b_f = b_f
        self.b_g = b_g
        self.bfunc = bfunc
        self.binary = binary
        self.use_r = use_r
        self.idxs_gauss = idxs_gauss
        self.idxs_delta = idxs_delta

    def fit(self, x, lsk):
        n, dim = x.shape
        if self.bfunc == 'linear':
            self.n_basis_ = {'f': dim+1, 'g': dim+1}
        if self.bfunc in ['gauss', 'gauss_plus_delta', 'gauss_times_delta']:
            self.n_basis_ = {
                'f': np.min((self.b_f, n)),
                'g': np.min((self.b_g, n))
            }
            ids_f = np.random.permutation(n)
            ids_f = ids_f[:self.n_basis_['f']]
            #ids_g = np.random.permutation(n)
            #ids_g = ids_g[:self.n_basis_['g']]
            ids_g = ids_f
            self.v_ = {
                'f': x[ids_f, :],
                'g': x[ids_g, :]
            }
        elif self.bfunc == 'polynomial':
            self.n_basis_ = {
                'f': 21,
                'g': 21
            }
            self.poly_ = PolynomialFeatures(2)

        if self.use_r:
            self.fit_with_r(x, lsk)
        else:
            xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)
            my = np.array([np.count_nonzero(ky == kk) for kk in ky], dtype=np.int32)
            nt = np.array([np.count_nonzero(kt == kk) for kk in kt], dtype=np.int32)
            self.fit_y_t_k(x, xy, y, ky, my, xt, t, kt, nt)

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        if not set(t).issubset([0, 1]):
            raise ValueError('t must be in {0, 1}')
        if not set(ky).issubset([-1, 1]):
            raise ValueError('ky must be in {0, 1}')
        if not set(kt).issubset([-1, 1]):
            raise ValueError('kt must be in {0, 1}')
        t = 2 * t - 1  # {0, 1} => {-1, +1}

        z = ky * y
        z = z[:, None]
        w = kt * t
        w = w[:, None]
        xz = xy
        xw = xt
        del xy, xt

        phi_w = self.feature(xw, 'f')
        phi_z = self.feature(xz, 'f')
        psi_w = self.feature(xw, 'g')
        psi_z = self.feature(xz, 'g')

        A = (w * phi_w).T.dot(psi_w) / w.shape[0]
        b = np.mean(z * psi_z, axis=0)
        Ctilde = (psi_w.T.dot(psi_w) / w.shape[0] + psi_z.T.dot(psi_z) / z.shape[0]) / 2 + self.reg_level_g * np.eye(self.n_basis_['g'])
        invC = np.linalg.inv(Ctilde)

        if self.binary:
            self.alpha_ = 2 * self.qp_cvxpy(phi=np.r_[phi_z, phi_w], A=A+self.reg_level_f, invC=invC, b=b)
        else:
            self.alpha_ = 2 * np.linalg.solve(A.dot(invC).dot(A.T) + self.reg_level_f * np.eye(A.shape[0]), A.dot(invC).dot(b))

        self.beta_ = invC.dot(A.T.dot(self.alpha_) - 2 * b)

        return self

    def fit_with_r(self, x, lskr):
        xy, y, ky, xt, t, kt, ry, rt = separate_xlskr(x, lskr)

        if not set(t).issubset([0, 1]):
            raise ValueError('t must be in {0, 1}')
        if not set(ky).issubset([-1, 1]):
            raise ValueError('ky must be in {0, 1}')
        if not set(kt).issubset([-1, 1]):
            raise ValueError('kt must be in {0, 1}')

        t = 2 * t - 1  # {0, 1} => {-1, +1}

        z = ky * y
        z = z[:, np.newaxis]
        w = kt * t
        w = w[:, np.newaxis]
        xz = xy
        xw = xt
        rz = ry[:, np.newaxis]
        rw = rt[:, np.newaxis]
        del xy, xt, ry, rt

        phi_w = self.feature(xw, 'f')
        psi_w = self.feature(xw, 'g')
        psi_z = self.feature(xz, 'g')

        A = (rw * w * phi_w).T.dot(psi_w) / w.shape[0]
        b = np.mean(rz * z * psi_z, axis=0)
        C = ((rw * psi_w).T.dot(psi_w) / w.shape[0] + (rz * psi_z).T.dot(psi_z) / z.shape[0]) / 2
        invC = np.linalg.inv(C + self.reg_level_g * np.eye(C.shape[0]))
        self.alpha_ = 2 * np.linalg.solve(A.dot(invC).dot(A.T) + self.reg_level_f * np.eye(A.shape[0]), A.dot(invC).dot(b))
        self.beta_ = invC.dot(A.T.dot(self.alpha_) - 2 * b)

        return self

    def qp_cvxpy(self, phi, A, invC, b):
        D = sp.linalg.sqrtm(invC)
        alpha = cvx.Variable(self.n_basis_['f'])

        #objective = cvx.Minimize(alpha * A * invC * A.T * alpha - 2 * b * invC * A.T * alpha)
        objective = cvx.Minimize(cvx.sum_squares(D.dot(A.T) * alpha - 2 * D.dot(b)))

        f = phi * alpha
        constraints = [-1 <= f, f <= 1]

        prob = cvx.Problem(objective, constraints)
        prob.solve()
        return alpha.value

    def feature(self, x, f_or_g):
        if self.bfunc == 'linear':
            return np.c_[x, 1E+10*np.ones(x.shape[0])]
        elif self.bfunc == 'gauss':
            v = self.v_[f_or_g]
            n_basis = self.n_basis_[f_or_g]
            band_width = {'f': self.band_width_f, 'g': self.band_width_g}[f_or_g]
            n, dim = x.shape
            vx = np.dot(x, v.T)
            xx = np.tile(np.sum(x ** 2, axis=1), (n_basis, 1))
            vv = np.tile(np.sum(v ** 2, axis=1), (n, 1))
            distmat = xx.T - 2 * vx + vv
            phi = np.exp(- distmat / band_width)
            return phi
            # return np.c_[phi, 1E+10 * np.ones(x.shape[0])]
        elif self.bfunc == 'polynomial':
            phi = self.poly_.fit_transform(x)
            return phi
        elif self.bfunc == 'gauss_plus_delta':
            v = self.v_[f_or_g]
            band_width = self.band_width_f if f_or_g == 'f' else self.band_width_g
            phi_gau = self.gaussian_kernel(
                x[:, self.idxs_gauss],
                v[:, self.idxs_gauss],
                band_width
            )
            phi_dlt = self.delta_kernel(
                x[:, self.idxs_delta],
                v[:, self.idxs_delta]
            )
            return phi_gau + phi_dlt
        elif self.bfunc == 'gauss_times_delta':
            v = self.v_[f_or_g]
            band_width = self.band_width_f if f_or_g == 'f' else self.band_width_g
            phi_gau = self.gaussian_kernel(
                x[:, self.idxs_gauss],
                v[:, self.idxs_gauss],
                band_width
            )
            phi_dlt = self.delta_kernel(
                x[:, self.idxs_delta],
                v[:, self.idxs_delta]
            )
            return phi_gau * phi_dlt

    @staticmethod
    def gaussian_kernel(x, v, band_width):
        vx = np.dot(x, v.T)
        xx = np.tile(np.sum(x ** 2, axis=1), (v.shape[0], 1))
        vv = np.tile(np.sum(v ** 2, axis=1), (x.shape[0], 1))
        distmat = xx.T - 2 * vx + vv
        #vv = np.sum(v ** 2, axis=1)
        #distmat = xx[:, np.newaxis] - 2 * vx[:, np.newaxis] + vv[np.newaxis, :]

        phi = np.exp(- distmat / band_width)
        return phi

    @staticmethod
    def delta_kernel(x, v):
        vx = np.dot(x, v.T)
        xx = np.tile(np.sum(x ** 2, axis=1), (v.shape[0], 1))
        vv = np.tile(np.sum(v ** 2, axis=1), (x.shape[0], 1))
        distmat = xx.T - 2 * vx + vv
        ## This is extremely slow (I don't know why):
        #xx = np.sum(x ** 2, axis=1)
        #vv = np.sum(v ** 2, axis=1)
        #distmat = xx[:, np.newaxis] - 2 * vx[:, np.newaxis] + vv[np.newaxis, :]

        phi = np.exp(- 10 * distmat)
        print(np.max(phi.ravel()))
        print(np.min(phi.ravel()))
        return phi

    def ranking_score(self, x):
        return np.dot(self.feature(x, 'f'), self.alpha_)

    def objective(self, x, lsk, add_reg=False):
        if self.use_r:
            xy, y, ky, xt, t, kt, ry, rt = separate_xlskr(x, lsk)

            z = ky * y
            z = z[:, np.newaxis]
            w = kt * t
            w = w[:, np.newaxis]
            xz = xy
            xw = xt
            rz = ry[:, np.newaxis]
            rw = rt[:, np.newaxis]
            del xy, xt, ry, rt

            phi_w = self.feature(xw, 'f')
            psi_w = self.feature(xw, 'g')
            psi_z = self.feature(xz, 'g')

            A = (rw * w * phi_w).T.dot(psi_w) / w.shape[0]
            b = np.mean(rz * z * psi_z, axis=0)
            C = ((rw * psi_w).T.dot(psi_w) / w.shape[0] + (rz * psi_z).T.dot(psi_z) / z.shape[0]) / 2

            #return 2 * self.alpha_.T.dot(A).dot(self.beta_) - 4 * b.T.dot(self.beta_) - self.beta_.T.dot(C).dot(self.beta_)
            return obj_helper(
                alpha=self.alpha_,
                beta=self.beta_,
                A=A,
                b=b,
                C=C
            )

        else:
            xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)

            z = ky * y
            z = z[:, None]
            w = kt * t
            w = w[:, None]
            xz = xy
            xw = xt
            del xy, xt

            phi_w = self.feature(xw, 'f')
            psi_w = self.feature(xw, 'g')
            psi_z = self.feature(xz, 'g')

            A = (w * phi_w).T.dot(psi_w) / w.shape[0]
            b = np.mean(z * psi_z, axis=0)
            C = (psi_w.T.dot(psi_w) / w.shape[0] + psi_z.T.dot(psi_z) / z.shape[0]) / 2

            return obj_helper(
                alpha=self.alpha_,
                beta=self.beta_,
                A=A,
                b=b,
                C=C
            )
            # return 2 * self.alpha_.T.dot(A).dot(self.beta_) - 4 * b.T.dot(self.beta_) - self.beta_.T.dot(C).dot(self.beta_)


def obj_helper(alpha, beta, A, b, C):
    return 2 * alpha.T.dot(A).dot(beta) - 4 * b.T.dot(beta) - beta.T.dot(C).dot(beta)
