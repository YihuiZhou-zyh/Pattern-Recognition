from numpy.linalg import inv
import numpy as np
from playML.metrics import accuracy_score


class LDA:
    def __init__(self, priors=[0.5, 0.5]):
        """init the LDA model"""
        """between-class scatter matrix"""
        self._Sb = None
        """pooled within-class scatter matrix"""
        self._Sw = None
        """各类的均值"""
        self._mean = None
        """各类的离散度矩阵    样本协防差"""
        self._cov = None
        """权向量"""
        self._w = None
        """w0"""
        self._w0 = None
        """先验概率"""
        self._priors = priors

    def fit(self, X_train, y_train):
        X_train, y_train = np.array(X_train), np.array(y_train)
        sorted_y = np.sort(np.unique(y_train))
        self._mean = np.array([X_train[y_train == y].mean(axis=0) for y in sorted_y])
        self._cov = np.array([np.cov(X_train[y_train == y].T) for y in sorted_y])
        self._Sw = sum(self._cov)
        mean_minus = np.array(self._mean[0] - self._mean[1])
        mean_add = np.array(self._mean[0] + self._mean[1])
        self._Sb = np.matmul(mean_minus, mean_minus.T)
        inv_Sw = inv(self._Sw)
        self._w = np.matmul(inv_Sw, mean_minus)
        self._w0 = -1 / 2 * np.matmul(mean_add.T, self._w) - np.log(self._priors[0] / self._priors[1])
        return self

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, X_test):
        return np.array([self._g(X_test) < 0], dtype="float64")

    def _g(self, X_test):
        return np.matmul(self._w.T, X_test) + self._w0

    def decision_function(self, X_test):
        b = [self._g(X_test[i]) for i in range(len(X_test))]
        return b

    def score(self, X_test, y_test):
        """根据数据集X_test，y_test计算准确度 分类准确度使用accuracy_score"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict.reshape(-1))
