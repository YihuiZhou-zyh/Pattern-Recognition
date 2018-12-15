import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集获得方差和均值"""
        assert X.ndim == 2,"the dimension of X must be 2"
        self.mean_ = np.mean(X)
        self.scale_ = np.std(X)

        return self

    def transform(self,X):
        assert X.ndim == 2, "the dimension of X must be 2"
        assert self.scale_ is not None and self.mean_ is not None,\
            "must fit before transform"
        resX = np.empty(shape=X.shape, dtype =float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
            return resX


