import numpy as np
import matplotlib.pyplot as plt


def v(x1, x2):
    return np.array([x1, x2])


w = 0.9492 * v(0.91, 0.32) + 0.3030 * (-1) * v(2.05, 1.54) + 0.9053 * (-1) * v(2.34, 0.72)
print "w=", w

b = (1 - 1 - 1 - w.dot(v(0.91, 0.32) + v(2.05, 1.54) + v(2.34, 0.72))) / 3
print b


def y(x):
    return (-1 * b - x * w[0]) / w[1]


def y_array(x_array):
    return [y(x) for x in x_array]


if __name__ == '__main__':
    x = np.arange(-2, 5, 0.1)
    y_array = y_array(x)
    plt.plot(x, y_array)
    plt.plot(
        [0.52, 0.91, -1.48, 0.01, -0.46, 0.41, 0.53, -1.21, -0.39, -0.96],
        [-1, 0.32, 1.23, 1.44, -0.37, 2.04, 0.77, -1.1, 0.96, 0.08],
        'ro'
    )
    plt.plot(
        [2.46, 3.05, 2.2, 1.89, 4.51, 3.06, 3.16, 2.05, 2.34, 2.96],
        [2.59, 2.87, 3.04, 2.64, -0.52, 1.3, -0.56, 1.54, 0.72, 0.13],
        'bo'
    )
    plt.show()


