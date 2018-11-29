import numpy as np
from sklearn.base import BaseEstimator

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import Chain
from chainer import cuda


class Sklearnify(BaseEstimator):
    def __init__(self
            , model
            , batch_size=128
            , n_epochs=1000
            , debug=False
            , lr=0.0001
            , validation_iter=None
            , reg_level=0.0005):
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.debug = debug
        self.validation_iter = validation_iter
        self.lr = lr
        self.reg_level = reg_level

    def fit(self, x, y):
        x = x.astype(np.float32)
        y = (1 + y) / 2
        y = y.astype(np.int32)

        train = chainer.datasets.tuple_dataset.TupleDataset(x, y)
        train_iter = iterators.SerialIterator(train, batch_size=self.batch_size)
        optimizer = optimizers.SGD(lr=self.lr)
        # optimizer = optimizers.AdaDelta(rho=0.999)
        optimizer.setup(self.model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(self.reg_level))
        # opt.add_hook(chainer.optimizer.GradientClipping(threshold=1))
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (self.n_epochs, 'epoch'), out='result')

        if self.debug:
            if self.validation_iter:
                trainer.extend(extensions.Evaluator(self.validation_iter, self.model))
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
            trainer.extend(extensions.ProgressBar())

        trainer.run()

        return self

    def predict(self, x):
        # yhat = np.argmax(self.decision_function(x), axis=1)

        x = x.astype(np.float32)
        yhat = np.argmax(self.model.predictor(x).data, axis=1)

        return yhat

    def decision_function(self, x):
        x = x.astype(np.float32)
        h_out = self.model.predictor(x)
        phat = F.softmax(h_out).data
        return phat[:, 0] - 0.5


class SklearnifyRegressor(BaseEstimator):
    def __init__(self
            , model
            , batch_size=128
            , n_epochs=1000
            , debug=False
            , lr=0.0001
            , validation_iter=None
            , reg_level=0.0005):
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.debug = debug
        self.validation_iter = validation_iter
        self.lr = lr
        self.reg_level = reg_level

    def fit(self, x, y):
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        train = chainer.datasets.tuple_dataset.TupleDataset(x, y)
        train_iter = iterators.SerialIterator(train, batch_size=self.batch_size)
        optimizer = optimizers.SGD(lr=self.lr)
        # optimizer = optimizers.AdaDelta(rho=0.999)
        optimizer.setup(self.model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(self.reg_level))
        # opt.add_hook(chainer.optimizer.GradientClipping(threshold=1))
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (self.n_epochs, 'epoch'), out='result')

        if self.debug:
            if self.validation_iter:
                trainer.extend(extensions.Evaluator(self.validation_iter, self.model))
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
            trainer.extend(extensions.ProgressBar())

        trainer.run()

        return self

    def predict(self, x):
        x = x.astype(np.float32)
        yhat = self.model.predictor(x).data[:, 0]
        return yhat


class NN(Chain):
    def __init__(self, n_mid_units=10, n_out=2):
        super(NN, self).__init__()
        with self.init_scope():
            self.l_in = L.Linear(in_size=None, out_size=n_mid_units)
            self.l_mid1 = L.Linear(in_size=n_mid_units, out_size=n_mid_units)
            self.l_out = L.Linear(in_size=n_mid_units, out_size=n_out)

    def __call__(self, x):
        h = F.relu(self.l_in(x))
        h = F.relu(self.l_mid1(h))
        return self.l_out(h)


class NNBNorm(Chain):
    def __init__(self, n_mid_units=10, n_out=2):
        super(NN, self).__init__()
        with self.init_scope():
            self.l_in = L.Linear(in_size=None, out_size=n_mid_units)
            self.bnorm1 = L.BatchNormalization(n_mid_units)
            self.l_mid1 = L.Linear(in_size=n_mid_units, out_size=n_mid_units)
            self.bnorm2 = L.BatchNormalization(n_mid_units)
            self.l_out = L.Linear(in_size=n_mid_units, out_size=n_out)

    def __call__(self, x):
        h = self.l_in(x)
        h = self.bnorm1(h)
        h = F.relu(h)
        h = self.l_mid1(h)
        h = self.bnorm2(h)
        h = F.relu(h)
        return self.l_out(h)

