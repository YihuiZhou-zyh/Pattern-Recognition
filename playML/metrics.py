from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from playML.utils import make_meshgrid, plot_contours, TP, FN, FP, TN, FPR, TPR, split_one


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


def plot_decision_boundary(ax, model, X, h=.02):
    xx, yy = make_meshgrid(X[:, 0], X[:, 1], h)
    # ax.xlim(xx.min(), xx.max())
    # ax.ylim(yy.min(), yy.max())
    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)


def plot_con_surface(clf, X, Y, labels, h=0.2):
    xx, yy = make_meshgrid(X[:, 0], X[:, 1], h)
    w = clf._w
    w0 = clf._w0
    Z = -(w0 + w[0] * xx + w[1] * yy) / w[2]
    fig = plt.figure()
    ax = Axes3D(fig)
    out = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    ax.set_xlabel(labels[0])
    #     ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel(labels[1])
    #     ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel(labels[2])
    #     ax.w_zaxis.set_ticklabels([])
    plt.show()


def plot_decision_boundary_svm(ax, svm, X):
    """this plots the boundary line plus the margin of the svm model"""
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, svm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    w = svm.coef_[0]
    b = svm.intercept_[0]
    # w0 * x0 + w1 * x1 + b = 0
    # => x1 = (-w0 * x0 - b) / w1
    up_y = -w[0] / w[1] * xx - b / w[1] + 1 / w[1]
    down_y = -w[0] / w[1] * xx - b / w[1] - 1 / w[1]
    up_index = ((up_y >= yy.min()) & (up_y <= yy.max()))
    down_index = ((down_y >= yy.min()) & (down_y <= yy.max()))
    plt.plot(xx[up_index], up_y[up_index])
    plt.plot(xx[down_index], down_y[down_index])
    plt.show()


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
    """calculate confusion _matrix between y_predict and y_true"""
    return np.array([
        [TN(y_predict, y_true), FP(y_predict, y_true)],
        [FN(y_predict, y_true), TP(y_predict, y_true)]
    ])


def precision_score(y_predict, y_true):
    """calculate the precision score between y_predict and y_true"""
    try:
        return TP(y_predict, y_true) / (TP(y_predict, y_true) + FP(y_predict, y_true))
    except:
        return 0.


def recall_score(y_predict, y_true):
    """calculate the recall score between y_predict and y_true"""
    try:
        return TP(y_predict, y_true) / (TP(y_predict, y_true) + FN(y_predict, y_true))
    except:
        return 0.


def roc(decision_scores, y_test):
    """bug has not been solved"""
    fprs = []
    tprs = []
    thresholds = np.arange(np.min(decision_scores), np.max(decision_scores))
    for t in thresholds:
        y_pred = np.array(decision_scores >= t, dtype=int)
        fprs.append(FPR(y_pred, y_test))
        tprs.append(TPR(y_pred, y_test))
    return fprs, tprs, thresholds


def LeaveOneOut(X, y, clf):
    """
    留一法
    returns accuracy of the method
    """
    score = 0
    for i in range(len(X)):
        X_train, y_train, x_test, y_test = split_one(X, y, i)
        clf.fit(X_train, y_train)
        score += clf.score(x_test.reshape(1, -1), y_test.reshape(1, -1))

    return score / len(y)
