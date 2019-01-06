import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def split_one(X, y, i):
    """forge one element as a test sample
    Returns
    -------
    X_train, y_train, x_test, y_test
    """
    index = np.ones(len(X), dtype=np.bool)
    index[i] = False
    x_test, y_test = X[i], y[i]
    return X[index], y[index], x_test, y_test


def make_meshgrid(x, y, h=0.2):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    # ax.clabel(out, inline=True, fontsize=12)
    return out


def FPR(y_predict, y_true):
    try:
        return FP(y_predict, y_true) / (TN(y_predict, y_true) + FP(y_predict, y_true))
    except:
        return 0.


def TPR(y_predict, y_true):
    try:
        return TP(y_predict, y_true) / (TP(y_predict, y_true) + FN(y_predict, y_true))
    except:
        return 0.


def FP(y_predict, y_true):
    return sum((y_predict == 0) & (y_true == 1))


def TN(y_predict, y_true):
    return sum((y_predict == 0) & (y_true == 0))


def FN(y_predict, y_true):
    return sum((y_predict == 1) & (y_true == 0))


def TP(y_predict, y_true):
    return sum((y_predict == 1) & (y_true == 1))
