import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators
from chainer import optimizers

import matplotlib.pyplot as plt

from ._utils import UpliftSepMixin
from ._utils import separate_xlsk
from ._neural_nets_util import NN


class MinMaxNNwithLin(UpliftSepMixin):
    def __init__(self,
                 n_epochs=1000,
                 batch_size=128,
                 reg_level=0.0005,
                 lr=0.0001,
                 debug=True,
                 validation_iter=None,
                 n_hidden=5,
                 band_width=5,
                 n_b=1000,
                 bfunc='gauss'
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_level=reg_level
        self.lr = lr
        self.debug = debug
        self.validation_iter = validation_iter
        self.n_hidden = n_hidden
        self.band_width = band_width
        self.n_b = n_b
        self.bfunc = bfunc

    def fit(self, x, lsk):
        x = x.astype(np.float32)
        lsk = lsk.astype(np.int32)  # TODO: l can be float for regression

        # Select anchor points for Gaussian basis functions.
        n = x.shape[0]
        self.n_basis_ = np.min((self.n_b, n))
        ids = np.random.permutation(n)
        ids = ids[:self.n_basis_]
        self.v_ = x[ids, :]

        train = chainer.datasets.tuple_dataset.TupleDataset(x, lsk)
        train_iter = iterators.SerialIterator(train, batch_size=self.batch_size, shuffle=True)

        self.model_ = NN(n_mid_units=self.n_hidden, n_out=1)

        opt = optimizers.SGD(lr=self.lr)
        opt.setup(self.model_)
        opt.add_hook(chainer.optimizer.WeightDecay(self.reg_level))
        # opt.add_hook(chainer.optimizer.GradientClipping(threshold=1))

        self.calc_b(x, lsk)
        self.calc_invC(x, lsk)

        loss_history = []
        if self.debug:
            i_epoch = 0
            loss_history = []
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        while train_iter.epoch < self.n_epochs:
            train_batch = train_iter.next()
            x_bat, lsk_bat = chainer.dataset.concat_examples(train_batch)

            loss_pa, loss = self.lossfun(x_bat, lsk_bat)
            loss_history.append(loss.data)
            if self.debug:
                loss_history.append(loss.data)
                i_epoch += 1
            self.model_.cleargrads()
            loss_pa.backward()
            if loss_pa.data < 0:
                opt.lr = -0.5*self.lr
                opt.update()
                opt.lr = self.lr
            else:
                opt.update()
            # Debug.print(loss.data, 'loss')
            if train_iter.is_new_epoch:
                if self.debug:
                    plt.sca(ax1)
                    plt.cla()
                    plt.sca(ax2)
                    plt.cla()
                    ax1.plot(range(i_epoch), loss_history)
                    if i_epoch > 100:
                        ax2.plot(range(100), loss_history[-100:])
                    plt.pause(0.05)
                print('epoch:{:02d} train_loss:{:.04f} '.format(
                    train_iter.epoch, float(loss_pa.data)), end='')

        print('')
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(loss_history[:100])
        ax2.plot(loss_history[100:1000])
        ax3.plot(loss_history[1000:2000])
        ax4.plot(loss_history[2000:])
        plt.show()
        if self.debug:
            print('')
            plt.show()

        return self

    def calc_b(self, x, lsk):
        xz, y, ky, xw, t, kt = separate_xlsk(x, lsk)
        z = ky * y
        z = z[:, None]

        psi_z = self.phi(xz)
        self.b = np.mean(z * psi_z, axis=0)[:, np.newaxis]

    def calc_invC(self, x, lsk):
        xz, y, ky, xw, t, kt = separate_xlsk(x, lsk)
        z = ky * y
        z = z[:, None]
        w = kt * t
        w = w[:, None]

        psi_w = self.phi(xw)
        psi_z = self.phi(xz)

        C = (psi_w.T.dot(psi_w) / w.shape[0] + psi_z.T.dot(psi_z) / z.shape[0]) / 2
        self.invC = np.linalg.inv(C + self.reg_level * np.eye(C.shape[0]))
        # self.invC = (psi_w.T.dot(psi_w) / w.shape[0] + psi_z.T.dot(psi_z) / z.shape[0]) / 2

    def lossfun(self, x, lsk):
        x = x.astype(np.float32)
        lsk = lsk.astype(np.int32)  # TODO: l can be float for regression
        xz, y, ky, xw, t, kt = separate_xlsk(x, lsk)

        w = kt * t
        w = w[:, None]

        nw = w.shape[0]
        nw1 = nw // 2
        nw2 = nw - nw1
        order = np.random.permutation(nw)
        order1 = order[:nw1]
        order2 = order[nw1:]

        w1 = w[order1, :]
        w2 = w[order2, :]

        psi_w = self.phi(xw).astype(np.float32)
        psi_w1 = psi_w[order1, :]
        psi_w2 = psi_w[order2, :]

        nu_w = self.model_(xw)
        nu_w1 = nu_w[order1, :]
        nu_w2 = nu_w[order2, :]

        # ath1 = F.mean(psi_w1 * w1 * F.tile(nu_w1, (1, self.n_b)), axis=0)[:, np.newaxis]
        # beta1 = F.mean(self.invC * F.tile(ath1 - self.b, (1, self.n_b)), axis=0)[np.newaxis, :]
        # # g = psi_w2.dot(beta)[:, np.newaxis]
        # g1 = F.mean(psi_w2 * F.tile(beta1, (nw2, 1)), axis=1)[:, np.newaxis]
        # # g = self.phi(xw).dot(self.invC).dot(ath - b)

        # ath = F.matmul(psi_w1 * w1, nu_w1, transa=True)
        ath = F.matmul(psi_w1 * w1, nu_w1, transa=True) / nw1
        beta = F.matmul(self.invC, ath - self.b)
        g = F.matmul(psi_w2, beta)
        g = F.cast(g, 'float32')

        # loss = 2 * F.matmul(w2 * nu_w2, g, transa=True)
        loss = 2 * F.matmul(w2 * nu_w2, g, transa=True) / nw2
        loss = loss[0, 0]
        # Passive-agressive heuristic:
        loss_pa = loss + 0.1 * F.sum(F.square(nu_w-nu_w.data.astype(np.float32))) / nw
        return loss_pa, loss # loss.shape == (1, 1)

    def ranking_score(self, x):
        x = x.astype(np.float32)
        uhat = self.model_(x)
        return uhat.data

    def phi(self, x):
        if self.bfunc == 'linear':
            self.n_basis_ = 3
            # return x
            return np.c_[x, 1E+10*np.ones(x.shape[0])]
        elif self.bfunc == 'gauss':
            n, dim = x.shape
            vx = np.dot(x, self.v_.T)
            xx = np.tile(np.sum(x ** 2, axis=1), (self.n_basis_, 1))
            vv = np.tile(np.sum(self.v_ ** 2, axis=1), (n, 1))
            distmat = xx.T - 2 * vx + vv
            phi = np.exp(- distmat / self.band_width)  # TODO: Remove 2
            return phi
        elif self.bfunc == 'polynomial':
            phi = self.poly.fit_transform(x)
            return phi

