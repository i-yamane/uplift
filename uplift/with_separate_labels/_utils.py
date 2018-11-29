import numpy as np
import inspect

import warnings
warnings.simplefilter('default')

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('Failed to import cvxpy. Do not enable `binary` option in MinMaxLinear.', ImportWarning)


class Debug:
    @staticmethod
    def print(*values, message=None, end='\n'):
        if message is not None:
            print(message, end=': ')

        print(', '.join([str(val) for val in values]), end=end)

    @staticmethod
    def print_var(*variables, message=None, end='\n'):
        if message is not None:
            print(message, end='> ')

        for var in variables:
            names = Debug.retrieve_name_val(var)
            print(', '.join([name + ': ' + str(var) for name in names]), end=end)

    @staticmethod
    def retrieve_name_val(var, trace_back=1):
        frame = inspect.currentframe().f_back
        for i in range(trace_back):
            frame = frame.f_back
        callers_local_vars = frame.f_locals.items()
        return [name for name, val in callers_local_vars if val is var]


def overrides(interface_class):
    """ mkorpela's overrides decorator.
    See https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
    """
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


class UpliftSepMixin:
    def fit(self, x, lsk):
        xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)

        my = np.array([np.sum(ky == kk) for kk in ky], dtype=np.int32)
        nt = np.array([np.sum(kt == kk) for kk in kt], dtype=np.int32)

        self.fit_y_t_k(x, xy, y, ky, my, xt, t, kt, nt)

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        raise NotImplementedError()

    def predict(self, x):
        z_hat = np.zeros((x.shape[0], 1), dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = self.ranking_score(x) > self.threshold
        z_hat[i_top] = 1

        return z_hat

    def rank(self, x):
        score = self.ranking_score(x)
        if len(score.shape) > 1 and score.shape[1] != 1:
            raise ValueError('A vector is expected, but a matrix is given.')
        if score.shape[0] != x.shape[0]:
            raise ValueError('Shapes do not match.')
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)

    def ranking_score(self, x):
        raise NotImplementedError()



def separate_xlsk(x, lsk):
    l, s, k = unpack_lsk(lsk)
    # Change coding from {0, 1} to {-1, +1} for l and k.
    #l = 2 * l - 1
    k = 2 * k - 1
    # Caution: s remains in {0, 1}!

    xy = x[s == 0, :]
    y = l[s == 0]
    ky = k[s == 0]

    xt = x[s == 1, :]
    t = l[s == 1]
    kt = k[s == 1]
    return xy, y, ky, xt, t, kt


def unpack_lsk(lsk):
    l = lsk[:, 0]
    s = lsk[:, 1]
    k = lsk[:, 2]

    #if not set(l).issubset({0, 1}):
    #    warnings.warn('Some labels have non-binary values: set={0}'.format(set(l)), RuntimeWarning)
    if not set(s).issubset({0, 1}):
        raise ValueError
    if not set(k).issubset({0, 1}):
        raise ValueError

    return l, s, k


def separate_xlskr(x, lskr):
    l, s, k, r = unpack_lskr(lskr)
    # Change coding from {0, 1} to {-1, +1} for l and k.
    #l = 2 * l - 1
    k = 2 * k - 1
    # Caution: s remains in {0, 1}!

    xy = x[s == 0, :]
    y = l[s == 0]
    ky = k[s == 0]
    ry = r[s == 0]

    xt = x[s == 1, :]
    t = l[s == 1]
    kt = k[s == 1]
    rt = r[s == 1]
    return xy, y, ky, xt, t, kt, ry, rt


def unpack_lskr(lskr):
    l = lskr[:, 0]
    s = lskr[:, 1]
    k = lskr[:, 2]
    r = lskr[:, 3]

    #if not set(l).issubset({0, 1}):
    #    warnings.warn('Some labels have non-binary values: set={0}'.format(set(l)), RuntimeWarning)
    if not set(s).issubset({0, 1}):
        raise ValueError
    if not set(k).issubset({0, 1}):
        raise ValueError

    return l, s, k, r

