import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Perceptron

data = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
target = np.array([0, 0, 0, 1])
p = Perceptron(max_iter=100)
p_out = p.fit(data, target)
# print(p.coef_, p.intercept_)

colors = np.array(['k', 'r'])
markers = np.array(['*', 'o'])

for data, target in zip(data, target):
    plt.scatter(data[0], data[1], s=100, c=colors[target], marker=markers[target])

grad = -p.coef_[0][0]/p.coef_[0][1]
intercept = -p.intercept_/p.coef_[0][1]

x_vals = np.linspace(0, 1)
y_vals = grad*x_vals + intercept
plt.plot(x_vals, y_vals)
plt.show()

