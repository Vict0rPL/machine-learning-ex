import pandas as pandas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

data = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w = np.zeros(1+x.shape[1])
        self.cost = []

        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = (y - output)
            self.w[1:] += self.eta * x.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def activaction(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.activaction(x) >= 0.0, 1, -1)


iris_target_values = data.iloc[0:100, 4].values
iris_target_values = np.where(iris_target_values == 'Iris-setosa', -1, 1)
iris_value_pairs = data.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(iris_value_pairs, iris_target_values)
ax[0].plot(range(1, len(ada1.cost) + 1), np.log10(ada1.cost), marker='o')
ax[0].set_xlabel('Periods')
ax[0].set_ylabel('Log(sum of the error squares)')
ax[0].set_title('Adaline - learning rate 0.001')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(iris_value_pairs, iris_target_values)
ax[1].plot(range(1, len(ada2.cost) + 1), np.log10(ada2.cost), marker='o')
ax[1].set_xlabel('Periods')
ax[1].set_ylabel('Log(sum of the error squares)')
ax[1].set_title('Adaline - learning rate 0.001')

plt.show()
