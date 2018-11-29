from chainer import links as L

from ._utils import UpliftSepMixin
from ._neural_nets_util import Sklearnify
from ._neural_nets_util import NN
from ._two_classifier_util import TwoClassifiers


class TwoNeuralNets(UpliftSepMixin):
    def __init__(self):
        pass

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        self.model_w_ = Sklearnify(L.Classifier(NN(n_mid_units=10, n_out=2)), lr=0.0001, n_epochs=1000, debug=True)
        self.model_z_ = Sklearnify(L.Classifier(NN(n_mid_units=10, n_out=2)), lr=0.0001, n_epochs=1000, debug=True)

        self._model_ = TwoClassifiers(model_w=self.model_w_, model_z=self.model_z_)
        self._model_.fit_y_t_k(x, xy, y, ky, my, xt, t, kt, nt)

        return self

    def ranking_score(self, x):
        return self._model_.ranking_score(x)

