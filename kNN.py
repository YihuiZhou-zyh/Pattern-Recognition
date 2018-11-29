import numpy as np
from math import sqrt
from collections import Counter


class kNN_classify:
    def __init__(self,k):
        """init kNN_classifier"""
        assert 1 <= k, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size  of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0],\
            "the size of X_train must be at least k"
        # assert X_train.shape[1] == x.shape[0],\
        #     "the feature number of x must be equal to X_train"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定带预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None,\
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待定数据x，返回x的预测结果值"""
        distances = [sqrt(np.sum(x_train - x)**2) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d" % self.k

