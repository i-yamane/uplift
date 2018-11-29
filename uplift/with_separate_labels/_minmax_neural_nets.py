import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators
from chainer import optimizers
from ._utils import UpliftSepMixin
from uplift.with_separate_labels import Debug
from ._utils import separate_xlsk
from ._neural_nets_util import NN

import matplotlib.pyplot as plt


class MinMaxNeuralNets(UpliftSepMixin):
    def __init__(self,
                 n_epochs=1000,
                 batch_size=128,
                 reg_level=0.0005,
                 lr_f=0.0001,
                 lr_g=0.0001,
                 debug=False,
                 validation_iter=None,
                 version=False,
                 n_hidden_f=5,
                 n_hidden_g=10,

    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_level=reg_level
        self.lr_f = lr_f
        self.lr_g = lr_g
        self.debug = debug
        self.validation_iter = validation_iter
        self.version = version
        self.n_hidden_f = n_hidden_f
        self.n_hidden_g = n_hidden_g

    def fit(self, x, lsk):
        x = x.astype(np.float32)
        lsk = lsk.astype(np.int32)
        train = chainer.datasets.tuple_dataset.TupleDataset(x, lsk)
        train_iter = iterators.SerialIterator(train, batch_size=self.batch_size, shuffle=True)
        self.model_f_ = NN(n_mid_units=self.n_hidden_f, n_out=1)
        self.model_g_ = NN(n_mid_units=self.n_hidden_g, n_out=1)
        if self.version == 'original':
            opt_g = optimizers.SGD(lr=-self.lr_g)
            opt_f = optimizers.SGD(lr=self.lr_f)
        elif self.version == 'sq':
            opt_g = optimizers.SGD(lr=-self.lr_g**2)
            opt_f = optimizers.SGD(lr=self.lr_f**2)
        elif self.version == 'heuristic':
            opt_g = optimizers.SGD(lr=self.lr_g)
            opt_f = optimizers.SGD(lr=self.lr_f)
        else:
            opt_g = optimizers.SGD(lr=self.lr_g)
            opt_f = optimizers.SGD(lr=self.lr_g)
        # opt_g = optimizers.Adam()
        # opt_f = optimizers.Adam()
        opt_f.setup(self.model_f_)
        opt_g.setup(self.model_g_)

        opt_f.add_hook(chainer.optimizer.WeightDecay(self.reg_level))
        opt_g.add_hook(chainer.optimizer.WeightDecay(self.reg_level))
        opt_f.add_hook(chainer.optimizer.GradientClipping(threshold=1))
        opt_g.add_hook(chainer.optimizer.GradientClipping(threshold=1))

        if self.debug:
            i_epoch = 0
            loss_history = []
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        while train_iter.epoch < self.n_epochs:
            train_batch = train_iter.next()
            x, lsk = chainer.dataset.concat_examples(train_batch)

            if self.version == 'original':
                loss = self.lossfun(x, lsk)
                self.model_f_.cleargrads()
                self.model_g_.cleargrads()
                loss.backward()
                if 0 < loss.data:
                    opt_f.update()
                opt_g.update()
                # Debug.print(loss.data, 'loss')
                if train_iter.is_new_epoch:
                    print('epoch:{:02d} train_loss:{:.04f} '.format(
                        train_iter.epoch, float(loss.data)), end='')
            elif self.version == 'heuristic':
                # raise NotImplementedError()
                loss_g = -self.lossfun(x, lsk)
                self.model_g_.cleargrads()
                loss_g.backward()
                opt_g.update()

                # loss_f = 0.1*self.loss_heuristic(x, lsk)
                loss_f = self.lossfun(x, lsk)
                self.model_f_.cleargrads()
                loss_f.backward()
                opt_f.update()
                # Debug.print(loss_f.data, 'loss_f')
                # Debug.print(-loss_g.data, 'loss_g')
            elif self.version == 'sq':
                loss = self.lossfun(x, lsk) ** 2
                self.model_f_.cleargrads()
                self.model_g_.cleargrads()
                loss.backward()
                opt_f.update()
                opt_g.update()
                Debug.print(loss.data, 'loss')
            else:
                for _ in range(5):
                    loss_g = -self.lossfun(x, lsk)
                    self.model_g_.cleargrads()
                    loss_g.backward()
                    opt_g.update()

                loss_f = self.lossfun(x, lsk)
                self.model_f_.cleargrads()
                loss_f.backward()
                opt_f.update()

            if self.debug:
                loss_history.append(loss.data)
                i_epoch += 1
                if train_iter.is_new_epoch:
                    plt.sca(ax1)
                    plt.cla()
                    plt.sca(ax2)
                    plt.cla()
                    ax1.plot(range(i_epoch), loss_history)
                    if i_epoch > 100:
                        ax2.plot(range(100), loss_history[-100:])
                    plt.pause(0.05)
                    print('epoch:{:02d} train_loss:{:.04f} '.format(
                        train_iter.epoch, float(loss.data)), end='')
        if self.debug:
            plt.show()

        return self

    def lossfun(self, x, lsk):
        xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)
        z = ky * y
        z = z[:, None]
        w = kt * t
        w = w[:, None]
        xz = xy
        xw = xt
        del xy, xt

        g_xw = self.model_g_(xw)
        g_xz = self.model_g_(xz)
        f_xw = self.model_f_(xw)
        f_xz = self.model_f_(xz)
        loss =\
            2 * F.mean(w * g_xw * f_xw)\
            - 2 * F.mean(z * g_xz)\
            - 0.5 * (F.mean(g_xw ** 2) + F.mean(g_xz ** 2))\
            - 0.01*F.mean((g_xw - g_xw.data)**2) - 0.01*F.mean((g_xz - g_xz.data)**2)

        # r = np.random.rand(1)[0]
        r = 1

        return r * loss

    def loss_heuristic(self, x, lsk):
        xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)
        z = ky * y
        z = z[:, None]
        w = kt * t
        w = w[:, None]
        xz = xy
        xw = xt
        del xy, xt

        g_xw = self.model_g_(xw)
        g_xz = self.model_g_(xz)
        f_xw = self.model_f_(xw)
        f_xz = self.model_f_(xz)
        loss = -2 * F.mean(w * g_xw * f_xw)\

        return loss

    def ranking_score(self, x):
        x = x.astype(np.float32)
        uhat = self.model_f_(x)
        return uhat.data


class MinMaxLayered(UpliftSepMixin):
    def __init__(self,
                 n_epochs=1000,
                 batch_size=128,
                 lr=0.0001,
                 debug=False,
                 validation_iter=None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.debug = debug
        self.validation_iter = validation_iter

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        raise NotImplementedError('Not necessary to be implemented.')

    def fit(self, x, lsk):
        x = x.astype(np.float32)
        lsk = lsk.astype(np.int32)
        train = chainer.datasets.tuple_dataset.TupleDataset(x, lsk)
        train_iter = iterators.SerialIterator(train, batch_size=self.batch_size, shuffle=True)

        # self.model_f_ = NNBNorm(n_mid_units=4, n_out=1)
        # self.model_w_ = NNBNorm(n_mid_units=7, n_out=1)
        # self.model_z_ = NNBNorm(n_mid_units=7, n_out=1)

        self.model_f_ = NN(n_mid_units=4, n_out=1)
        self.model_w_ = NN(n_mid_units=7, n_out=1)
        self.model_z_ = NN(n_mid_units=7, n_out=1)

        opt_f = optimizers.SGD(lr=self.lr)
        opt_w = optimizers.SGD(lr=-self.lr)
        opt_z = optimizers.SGD(lr=-self.lr)

        opt_f.setup(self.model_f_)
        opt_w.setup(self.model_w_)
        opt_z.setup(self.model_z_)

        opt_f.add_hook(chainer.optimizer.WeightDecay(0.0005))
        opt_w.add_hook(chainer.optimizer.WeightDecay(0.0005))
        opt_z.add_hook(chainer.optimizer.WeightDecay(0.0005))
        # opt_f.add_hook(chainer.optimizer.GradientClipping(threshold=100))

        while train_iter.epoch < self.n_epochs:
            train_batch = train_iter.next()
            x, lsk = chainer.dataset.concat_examples(train_batch)

            for _ in range(1):
                loss = self.lossfun(x, lsk)
                self.model_w_.cleargrads()
                self.model_z_.cleargrads()
                loss.backward()
                opt_w.update()
                opt_z.update()

            loss = self.lossfun(x, lsk)
            self.model_f_.cleargrads()
            loss.backward()
            opt_f.update()

            # self.model_f_.cleargrads()
            # self.model_w_.cleargrads()
            # self.model_z_.cleargrads()
            # loss.backward()
            # opt_f.update()
            # opt_w.update()
            # opt_z.update()

            if train_iter.is_new_epoch:
                print('epoch:{:02d} train_loss:{:.04f} '.format(
                    train_iter.epoch, float(loss.data)), end='')

        return self

    def g(self, x):
        return self.model_w_(x) * self.model_f_(x) - self.model_z_(x)

    def lossfun(self, x, lsk):
        xy, y, ky, xt, t, kt = separate_xlsk(x, lsk)
        z = ky * y
        z = z[:, None]
        w = kt * t
        w = w[:, None]
        xz = xy
        xw = xt
        del xy, xt

        g_xw = self.g(xw)
        g_xz = self.g(xz)
        f_xw = self.model_f_(xw)
        f_xz = self.model_f_(xz)
        loss =\
            2 * F.mean(w * g_xw * f_xw)\
            - 2 * F.mean(z * f_xz)\
            - 0.5 * (F.mean(g_xw ** 2) + F.mean(g_xz ** 2))

        return loss

    def ranking_score(self, x):
        x = x.astype(np.float32)
        uhat = self.model_f_(x)
        return uhat.data
