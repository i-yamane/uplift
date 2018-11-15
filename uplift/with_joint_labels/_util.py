import numpy as np
import matplotlib.pyplot as plt


def mean_err(z_true, z_hat):
    return np.mean(np.abs(z_true - z_hat))


def zero_one(z):
    if z.data > 0:
        zz = 1
    else:
        zz = 0
    return zz


def sym2numpy_func(args, expr, nin, nout):
    """Converts a SymPy expression into a NumPy lambda object that can be broad-casted over NumPy arrays.
    Example:
         func = sym2numpy_func((x, y), x+y, 2, 1)
         func(np.array([]))
    """
    flam = smp.lambdify(args, expr)
    return np.frompyfunc(flam, nin, nout)


def check_grad(func, grad, v0):
    eps = np.sqrt(np.finfo(float).eps)
    g_num = np.zeros(v0.shape)
    e = np.zeros(v0.shape)
    for ja in range(v0.shape[0]):
        for jb in range(v0.shape[1]):
            e[ja, jb] = 1
            g_num[ja, jb] = (func(v0+eps*e) - func(v0-eps*e)) / (2*eps)
            e[ja, jb] = 0
    return grad(v0) - g_num


def taylor_err(func, grad, v0):
    # eps = np.finfo(float).eps
    eps = np.sqrt(np.finfo(float).eps)
    f0, g0 = func(v0), grad(v0)
    df_approx, e = np.zeros(v0.shape), np.zeros(v0.shape)
    for ja in range(v0.shape[0]):
        for jb in range(v0.shape[1]):
            e[ja, jb] = 1
            dv = eps * e
            f1_approx = f0 + np.sum(g0 * dv)
            f1 = func(v0 + dv)
            df_approx[ja, jb] = f1 - f1_approx
            e[ja, jb] = 0
    return df_approx


class UpliftMixin:
    def fit(self, x=None, yt=None):
        y, t = unpack(yt)
        z = np.array(y == t, dtype=int)
        z = 2 * z - 1  # Change representation from {0, 1} to {-1, +1}
        self.fit_z(x, z)
        return self

    def fit_z(self, x, z):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def rank(self, x):
        score = self.ranking_score(x)
        i_sorted = np.argsort(a=score, axis=0)[::-1]
        return i_sorted.astype(int)

    def ranking_score(self, x):
        raise NotImplementedError()


class UpliftWrap:
    def __init__(self, score_func, n_treated=None, threshold=None):
        self.score_func = score_func
        self.n_treated = n_treated
        self.threshold = threshold

    def fit(self, x=None, yt=None):
        pass

    def fit_z(self, x=None, z=None):
        pass

    def predict(self, x):
        """Predicts z given x.
        """
        z_hat = np.zeros(x.shape[0], dtype=int)
        if self.n_treated is not None and self.threshold is None:
            i_top = self.rank(x)[:self.n_treated]
        elif self.threshold is not None and self.n_treated is None:
            i_top = np.array(self.ranking_score() > self.threshold, dtype=int)
        else:
            raise ValueError()

        z_hat[i_top] = 1

        return z_hat

    def rank(self, x):
        scores = self.score_func(x)
        i_ranked = np.argsort(a=scores, axis=0)[::-1]
        return i_ranked.astype(int)


def calc_AUUC(model, x, y, t, propensity):
    _, uplift = calc_uplift(model, x, y, t, propensity=propensity)
    return np.mean(uplift)


def max_uplift(model, x, y, t, propensity):
    prop_max, uplift = calc_uplift(model, x, y, t, propensity)
    return prop_max, np.max(uplift)


def calc_AUUC2(model, x, y, t, props=np.linspace(0, 1, 100)):
    _, uplift = calc_uplift(model, x, y, t, props)
    return np.mean(uplift)


def max_uplift2(model, x, y, t, props=np.linspace(0, 1, 100)):
    prop_max, uplift = calc_uplift(model, x, y, t, props)
    return prop_max, np.max(uplift)


def calc_actual_uplift(model, x, gen_y_given_x_ctl, gen_y_given_x_trt):
    n = x.shape[0]
    rank = model.rank(x)

    props, realuplift = [], []
    for n_selected in range(n + 1):
        y_ctl = gen_y_given_x_ctl(x)  # y ~ p_C(y | x) for each x
        y_trt = gen_y_given_x_trt(x)  # y ~ p_T(y | x) for each x
        props.append(n_selected)
        top_k = rank[0:n_selected]
        """ r_diff is equivalent to the following version:
        r_diff = r_after - r_before
        r_before = np.sum(y_ctl) / n
        r_after = (np.sum(y_ctl) - np.sum(y_ctl[top_k]) + np.sum(y_trt[top_k])) / n
        """
        r_diff = (np.sum(y_trt[top_k]) - np.sum(y_ctl[top_k])) / n
        realuplift.append(r_diff)

    return props, realuplift


def plot_uplift_curve(model, x_ctl, x_trt, y_ctl, y_trt, label='', color='k'):
    props, uplift = calc_uplift(model, x_ctl, x_trt, y_ctl, y_trt)
    plt.plot(props, uplift, label=label, color=color)


def plot_diff_prob(method, dim, ax):
    delta = 0.1
    x1 = np.arange(-10.0, 10.0, delta)
    x2 = np.arange(-10.0, 10.0, delta)
    X1, X2 = np.meshgrid(x1, x2)
    x = np.c_[X1.flatten(), X2.flatten()]
    n = x.shape[0]
    x = np.c_[x, np.zeros((n, dim - 2))]
    z = method.predict_average_uplift(x)
    Z = np.reshape(z, newshape=X1.shape)
    CS = ax.contour(X1, X2, Z, colors='k', levels=np.linspace(-1, 1, 11))
    ax.clabel(CS, inline=1, fontsize=10)


def unpack(yt):
    y = yt[:, 0]
    t = yt[:, 1]

    if not set(y).issubset({0, 1}):
        raise ValueError
    if not set(t).issubset({0, 1}):
        raise ValueError

    return y, t


def calc_uplift2(model, x, y, t, props=None):
    n_trt = np.sum(t)
    n_ctl = len(t) - n_trt

    x_ctl = x[t == 0, :]
    x_trt = x[t == 1, :]
    y_ctl = y[t == 0]
    y_trt = y[t == 1]

    # if props is None:
    #     Proportions of the treatment group
        # props = [k / n_trt for k in range(n_trt + 1)] + [k / n_ctl for k in range(n_ctl + 1)]
        # props = list(set(props))  # Unique list
        # props.sort()  # Sort *in place*

    nall = len(t)
    if props is None:
        props = [(k + 1) / nall for k in range(nall)]

    rank_ctl = model.rank(x_ctl)
    rank_trt = model.rank(x_trt)

    uplift = []
    for i, prop_tgt in enumerate(props):
        top_k_ctl = rank_ctl[0:int(prop_tgt * n_ctl)]
        top_k_trt = rank_trt[0:int(prop_tgt * n_trt)]
        if n_ctl == 0 or n_trt == 0:
            raise ValueError('Provide at least one sample in each of the control and the treatment sets.')

        r_ctl = np.sum(y_ctl[top_k_ctl]) / n_ctl
        r_trt = np.sum(y_trt[top_k_trt]) / n_trt
        # Note: These are not np.mean(y_...[top_k_...]).
        # The denominators are different.

        uplift.append(r_trt - r_ctl)

    return props, np.array(uplift)


def calc_uplift(model, x, y, t, propensity):
    n_trt = np.sum(t)
    n_ctl = len(t) - n_trt
    nall = len(t)

    #if propensity is None:
    #    propensity = np.ones(nall)

    x_ctl = x[t == 0, :]
    x_trt = x[t == 1, :]
    y_ctl = y[t == 0]
    y_trt = y[t == 1]
    prop_ctl = propensity[t == 0]
    prop_trt = propensity[t == 1]

    ths = [(k + 1) / nall for k in range(nall)]

    rank_ctl = model.rank(x_ctl)
    rank_trt = model.rank(x_trt)

    uplift = []
    for i, th in enumerate(ths):
        top_k_ctl = rank_ctl[0:int(th * n_ctl)]
        top_k_trt = rank_trt[0:int(th * n_trt)]

        epsilon = 1E-6

        r_ctl = np.sum(y_ctl[top_k_ctl]/(prop_ctl[top_k_ctl]+epsilon)) / nall
        r_trt = np.sum(y_trt[top_k_trt]/(prop_trt[top_k_trt]+epsilon)) / nall
        # E[Y*1[f(x) > f(x_kth)]/p(T=0|x) 1[T=0]]
        # E[Y*1[f(x) > f(x_kth)]/p(T=1|x) 1[T=1]]

        uplift.append(r_trt - r_ctl)

    return ths, np.array(uplift)

