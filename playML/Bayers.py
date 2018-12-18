import numpy as np
from numpy.linalg import inv

from .metrics import r2_score, accuracy_score


class BayersGN:
    def __init__(self, priors=[0.5, 0.5]):
        """初始化Linear Regression模型"""
        self.priors = priors
        self._mean = None
        self._cov = None

    def fit(self, X_train, y_train):
        """
        sorted_y 和priors对应的类别序号相对应
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        sorted_y = np.sort(np.unique(y_train))
        self._mean = [X_train[y_train == y].mean(axis=0) for y in sorted_y]
        self._cov = [np.cov(X_train[y_train == y].T) for y in sorted_y]
        return self

    def predict(self, X_predict):
        """给定带预测数据集X_predict,返回表示X_predict的结果向量"""
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, Xi):
        return np.argsort(self._rvs_g(Xi))[0]

    def decision_function(self, X_test):
        b = [self._rvs_g(X_test[i])[0]-self._rvs_g(X_test[i])[1] for i in range(len(X_test))]
        return b

    def score(self, X_test, y_test):
        """根据数据集X_test，y_test计算准确度 分类准确度使用accuracy_score"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def _rvs_g(self, Xi):
        if self._cov[0].ndim == 0:
            rvs_g = [(Xi - self._mean[i]) / self._cov[i] + np.log(self._cov[i]) - 2 * np.log(self.priors[i]) for i in
                     range(len(self._mean))]
        else:
            rvs_g = [np.matmul(np.matmul((Xi - self._mean[i]), inv(self._cov[i])),
                               (Xi - self._mean[i])) +
                     np.log(np.linalg.det(self._cov[i])) - 2 * np.log(self.priors[i]) for i in range(len(self._mean))]
        return rvs_g
