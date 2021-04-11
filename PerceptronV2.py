import pandas as pandas
import matplotlib.pyplot as pyplot
import numpy as numpy
from matplotlib.colors import ListedColormap

data = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


class Perceptron(object):
    def __init__(self, learning_rate=0.01, n_iter=10, random_seed=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.errors = []
        self.weights = []

    def fit(self, data_pairs, target_values):
        generate_random = numpy.random.RandomState(self.random_seed)
        self.weights = generate_random.normal(loc=0.0, scale=0.01, size=1 + data_pairs.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for data_pair, target in zip(data_pairs, target_values):
                update = self.learning_rate * (target - self.predict(data_pair))
                self.weights[1:] += update * data_pair
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, value_pairs):
        return numpy.dot(value_pairs, self.weights[1:]) + self.weights[0]

    def predict(self, value_pair):
        """Returns class label"""
        return numpy.where(self.net_input(value_pair) >= 0.0, 1, -1)


# define values used for learning
iris_target_values = data.iloc[0:100, 4].values
iris_target_values = numpy.where(iris_target_values == 'Iris-setosa', -1, 1)
iris_value_pairs = data.iloc[0:100, [0, 2]].values

# # Iris data visualization
# pyplot.scatter(iris_value_pairs[:50, 0], iris_value_pairs[:50, 1], color='red', marker='o', label='Setosa')
# pyplot.scatter(iris_value_pairs[50:100, 0], iris_value_pairs[50:100, 1], color='blue', marker='x', label='Versicolor')
# pyplot.xlabel('sepal length [cm]')
# pyplot.ylabel('petals length [cm]')
# pyplot.legend(loc='upper left')
# pyplot.show()

# # errors graph
perceptron = Perceptron(learning_rate=0.1, n_iter=10)
perceptron.fit(iris_value_pairs, iris_target_values)
# pyplot.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
# pyplot.xlabel('Iterations')
# pyplot.ylabel('Number of updates')
# pyplot.show()


def plot_decision_regions(value_pairs, target_values, classifier, resolution=0.02):
    # markers and color config
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(numpy.unique(target_values))])

    x1_min, x1_max = value_pairs[:, 0].min() - 1, value_pairs[:, 0].max() + 1
    x2_min, x2_max = value_pairs[:, 1].min() - 1, value_pairs[:, 1].max() + 1
    xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution), numpy.arange(x2_min, x2_max, resolution))

    labels = classifier.predict(numpy.array([xx1.ravel(), xx2.ravel()]).T)
    labels = labels.reshape(xx1.shape)

    pyplot.contourf(xx1, xx2, labels, alpha=0.4, cmap=cmap)
    pyplot.xlim(xx1.min(), xx1.max())
    pyplot.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(numpy.unique(target_values)):
        pyplot.scatter(x=value_pairs[target_values == cl, 0], y=value_pairs[target_values == cl, 1], alpha=0.8,
                       c=cmap(idx), marker=markers[idx], label=cl)


plot_decision_regions(iris_value_pairs, iris_target_values, classifier=perceptron)
pyplot.xlabel('sepal length')
pyplot.ylabel('petal length')
pyplot.legend(loc='upper left')
pyplot.show()
