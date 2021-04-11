import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_frame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


class Perceptron(object):
    """Classifier - perceptron
    Parameters
    ---------
    eta - learning rate (0.0 - 1.0)
    n_iter - number of iterations over learning data sets

    Attributes
    ----------
    w_ - one-dimensional table of weight after calculations
    errors_ - list, number of incorrect assignments of class
    """
    def __init__(self, eta=0.01, n_iter=10, random_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_seed = random_seed

    def fit(self, X, y):
        """Fit learning data
        Parameters
        ----------
        X - {array-like}, dimensions = [n_samples, n_traits]
         Learning vectors, where
         n_samples = number of samples
         n_traits = number of traits

        y - array-like, dimensions = [n_samples]
        target values

        Returns
        -------
        self: object
         """
        # self.w_ = np.zeros(1 + X.shape[1])
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print("Initial weights: ", self.w_)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        print("Final weights: ", self.w_)
        return self

    def net_input(self, X):
        """Calculates total stimulation"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Returns class label"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


y = data_frame.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = data_frame.iloc[0:100, [0, 2]].values

# # Iris data visualization
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
# plt.xlabel('leaves length [cm]')
# plt.ylabel('petals length [cm]')
# plt.legend(loc='upper left')
# plt.show()

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(X, y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Number of updates')
plt.show()

