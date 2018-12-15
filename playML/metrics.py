from math import sqrt

import numpy as np
import matplotlib.pyplot as plt


def accuracy_score(y_true, y_predict):
    """y_true and y_predict """
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_predict == y_true) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""
    return np.sum(np.absolute(y_predict - y_true) / len(y_true))


def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)


def plot_decision_boundary(ax, model, X, h=.02, y=None):
    def make_meshgrid(x, y, h=h):
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
        return out

    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)


def plot_decision_boundary_svm(ax, svm, X):
    def make_meshgrid(x, y, h=.02):
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
        w = clf.coef_[0]
        b = clf.intercept_[0]
        # w0 * x0 + w1 * x1 + b = 0
        # => x1 = (-w0 * x0 - b) / w1
        up_y = -w[0] / w[1] * xx - b / w[1] + 1 / w[1]
        down_y = -w[0] / w[1] * xx - b / w[1] - 1 / w[1]

        up_index = ((up_y >= yy.min()) & (up_y <= yy.max()))
        down_index = ((down_y >= yy.min()) & (down_y <= yy.max()))
        plt.plot(xx[up_index], up_y[up_index])
        plt.plot(xx[down_index], down_y[down_index])
        return out

    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, svm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)


def plot_learning_curve(model, X_train, y_train, X_test, y_test):
    test_err = []
    train_err = []
    for i in range(1, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        train_pre = model.predict(X_train[:i])
        train_err.append(mean_squared_error(train_pre, y_train[:i]))
        test_pre = model.predict(X_test)
        test_err.append(mean_squared_error(test_pre, y_test))

    plt.plot([i for i in range(1, len(X_train))], np.sqrt(train_err), label="train")
    plt.plot([i for i in range(1, len(X_train))], np.sqrt(test_err), label="test")
    # ymin = np.min(np.min(train_err), np.min(test_err)) - 1
    # ymax = np.max(np.max(train_err), np.max(test_err)) + 1
    plt.axis([0, len(X_train), 2, 10])
    plt.legend()
    plt.show()


def confusion_matrix(y_predict, y_true):
    return np.array([
        [TN(y_predict, y_true), FP(y_predict, y_true)],
        [FN(y_predict, y_true), TP(y_predict, y_true)]
    ])


def precision_score(y_predict, y_true):
    try:
        return TP(y_predict, y_true) / (TP(y_predict, y_true) + FP(y_predict, y_true))
    except:
        return 0.


def recall_score(y_predict, y_true):
    try:
        return TP(y_predict, y_true) / (TP(y_predict, y_true) + FN(y_predict, y_true))
    except:
        return 0.


def FPR(y_predict, y_true):
    try:
        return FP(y_predict, y_true) / (TN(y_predict, y_true) + FP(y_predict, y_true))
    except:
        return 0.


def TPR(y_predict, y_true):
    return recall_score(y_predict, y_true)


def FP(y_predict, y_true):
    return sum((y_predict == 0) & (y_true == 1))


def TN(y_predict, y_true):
    return sum((y_predict == 0) & (y_true == 0))


def FN(y_predict, y_true):
    return sum((y_predict == 1) & (y_true == 0))


def TP(y_predict, y_true):
    return sum((y_predict == 1) & (y_true == 1))


def roc(decision_scores, y_test):
    fprs = []
    tprs = []
    thresholds = np.arange(np.min(decision_scores), np.max(decision_scores))
    for t in thresholds:
        y_pred = np.array(decision_scores >= t, dtype=int)
        fprs.append(FPR(y_pred, y_test))
        tprs.append(TPR(y_pred, y_test))
    return fprs, tprs, thresholds
