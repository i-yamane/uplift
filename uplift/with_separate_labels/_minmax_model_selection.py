from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import numpy as np
import itertools as itt

import warnings
warnings.simplefilter('default')

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('Failed to import cvxpy. Do not enable `binary` option in MinMaxLinear.', ImportWarning)

#from bayes_opt import BayesianOptimization as BO
from ._utils import UpliftSepMixin


def calc_score_grid(estimator, x_train, lsk_train, x_valid, lsk_valid, min_params, max_params):
    """
    :param estimator: Expected to have these methods:
        * set_params(**params)
        * fit(x, lsk)
        * objective(x, lsk)
    :param x_train: Input data for training
    :param lsk_train: Labels for training
    :param x_valid: Input data for validation
    :param lsk_valid: Labels for validation
    :param min_params: List of maps from parameter names to values for the minimization variables.
    :param max_params: List of maps from parameter names to values for the maximization variables.
    :return: scores
    """

    scores = None
    for p_min in min_params:
        scores_row = None
        for p_max in max_params:
            params = merge_dict(p_min, p_max)
            estimator.set_params(**params)
            estimator.fit(x_train, lsk_train)
            score = estimator.objective(x_valid, lsk_valid)

            score = np.array([score])[:, np.newaxis]
            if scores_row is None:
                scores_row = score
            else:
                scores_row = np.concatenate((scores_row, score), axis=1)

        if scores is None:
            scores = scores_row
        else:
            scores = np.concatenate((scores, scores_row), axis=0)

    return scores


def minmax_grid(scores):
    scores_max = np.max(scores, axis=1)
    i_best = np.argmin(scores_max, axis=0)

    js_max = np.argmax(scores, axis=1)
    j_best = js_max[i_best]

    return i_best, j_best


class GridSearchMinMax(UpliftSepMixin):
    def __init__(self, estimator, min_params, max_params, k_cv):
        self.estimator = estimator
        self.params_grid_minpart = grid(min_params)
        self.params_grid_maxpart = grid(max_params)
        self.k_cv = k_cv

    def fit(self, x, lsk):
        scores = None
        self.folds_ = StratifiedKFold(n_splits=self.k_cv, shuffle=True)
        for index_train, index_valid in self.folds_.split(x, lsk[:, 1]):
            score = calc_score_grid(
                estimator=self.estimator,
                min_params=self.params_grid_minpart,
                max_params=self.params_grid_maxpart,
                x_train=x[index_train, :],
                lsk_train=lsk[index_train, :],
                x_valid=x[index_valid, :],
                lsk_valid=lsk[index_valid, :]
            )

            score = score[:, :, np.newaxis]
            if scores is None:
                scores = score
            else:
                scores = np.concatenate((scores, score), axis=2)

        avg_scores = np.mean(scores, axis=2)
        i_best, j_best = minmax_grid(avg_scores)
        self.best_params_ = merge_dict(self.params_grid_minpart[i_best], self.params_grid_maxpart[j_best])
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(x, lsk)
        print('Chosen parameters: ', self.best_params_)

        return self

    def fit_y_t_k(self, x, xy, y, ky, my, xt, t, kt, nt):
        raise AssertionError()

    def ranking_score(self, x):
        return self.estimator.ranking_score(x)


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def grid(params):
    """
    Return the product of lists stored in dictionaries.
    Example:
    >>> grid({'f': [1, 2], 'g': [3, 4]})
    ... [{'f': 1, 'g': 3}, {'f': 1, 'g': 4}, {'f': 2, 'g': 3}, {'f': 1, 'g': 4}]
    :param params: A map from a parameter name to list of values
    :return: List of maps from a parameter name to a value for all combinations of values in the input value list.
    """

    # Example: params == {'f': [1, 2], 'g': [3, 4]}).

    items = list(params.items())
    items.sort()  # `sort` is a bang method: it changes `items`.
    keys = [k for k, v in items]
    # Example: keys => ['f', 'g']

    val_lists = [v for k, v in items]
    # Example: val_lists => [[1, 2], [3, 4]]

    val_combs = list(itt.product(*val_lists))
    #: The generator is converted to a list so that it can be iterated over multiple times.
    # Example: val_combs => [(1, 3), (1, 4), (2, 3), (2, 4)]

    n_keys = len(keys)
    res = [{keys[i]: vals[i] for i in range(n_keys)} for vals in val_combs]
    # Example: res => [{'f': 1, 'g': 3}, {'f': 1, 'g': 4}, {'f': 2, 'g': 3}, {'f': 1, 'g': 4}]

    return res
