import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)


class Regression:
    def __init__(self, step = .01):
        self.params = None
        self.step = step
        self.X = None
        self.y = None
        self.t = 1e-5

    def desc(self):
        diff = np.dot(self.X, self.params) - self.y
        return np.dot(diff, self.X) / len(self.params) * self.step

    def gd(self):
        while True:
            desc = self.desc()
            if np.all(np.abs(desc)) < self.t:
                break
            self.params -= desc

    def fit(self, X, y):
        if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
            print('nan exists in X')
            return

        self.X = np.c_[X, np.ones((len(y), 1))]
        self.y = y
        self.params = np.zeros(self.X.shape[1])
        self.gd()

    def predict(self, X):
        X = np.c_[X, np.ones((len(X), 1))]
        return np.dot(X, self.params)


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = Regression()
    reg.fit(X, y)
    ny = reg.predict(X)
    print(reg.params)
    print(ny)