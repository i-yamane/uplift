from ._utils import UpliftSepMixin


class TwoClassifiers(UpliftSepMixin):
    def __init__(self, model_w, model_z):
        self.model_w = model_w
        self.model_z = model_z

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        z = ky * y
        w = kt * t

        xz = xy
        xw = xt
        del xy, xt

        self.model_w.fit(xw, w)
        self.model_z.fit(xz, z)

        return self

    def ranking_score(self, x):
        return self.model_z.decision_function(x) / self.model_w.decision_function(x)
