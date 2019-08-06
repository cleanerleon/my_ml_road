import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, step=.01):
        self.params = None
        self.step = step
        self.X = None
        self.y = None
        self.t = 1e-5

    def desc(self):
        y_hat = np.dot(self.X, self.params)
        yh_exp = np.exp(y_hat)
        desc = np.dot(self.y - yh_exp/(1+yh_exp), self.X)
        # desc = (self.X * self.y.reshape((-1,1))).sum(1) - yh_exp / ( 1 + yh_exp) * self.X.sum(1)
        return desc * self.step

    def gd(self):
        while True:
            desc = self.desc()
            if np.all(np.abs(desc) < self.t):
                break
            self.params += desc
            print(desc)

    def fit(self, X, y):
        if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
            print('nan exists in X')
            return

        self.X = np.c_[X, np.ones((len(y), 1))]
        self.y = y
        self.params = np.ones(self.X.shape[1])
        self.gd()

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0], 1))]
        z = np.dot(X, self.params)
        y = 1/1+np.exp(-z)
        print(y)
        y[y>=.5] = 1
        y[y<.5] = 0
        return y


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y[y<=1] = 0
    y[y>1] = 1
    clf = LogisticRegression()
    clf.fit(X, y)
    print(clf.predict(X))
