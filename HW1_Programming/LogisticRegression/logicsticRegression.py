import math
import numpy as np


def log(n):
    return math.log(n)


def exp(n):
    return math.exp(n)


def transpose(M):
    return np.transpose(M)


class logistic:
    def __init__(self, parameters):
        self.parameters = parameters
        self.N = len(parameters)
        assert(self.N == 3)
        self.X = np.array([
            [1, 60, 155],
            [1, 64, 135],
            [1, 73, 170]
        ])
        self.Y = transpose(np.array([0, 1, 1]))
        self.beta = transpose(np.array([self.parameters]))

    def log_likelihood(self):
        X, Y, beta = self.X, self.Y, self.beta
        ll = 0.0
        for i in range(self.N):
            xiT_beta = transpose(X[i]).dot(beta)
            ll += Y[i] * xiT_beta - log(1 + exp(xiT_beta))
        return ll

    def gradients(self):
        gradients = []
        return gradients

    def iterate(self):
        log_likelihood = self.log_likelihood()
        return self.parameters

    def hessian(self):
        n = len(self.parameters)
        hessian = np.zeros((n, n))
        return hessian


parameters = [0.25, 0.25, 0.25]
iterations = 2
for i in range(iterations):
    l = logistic(parameters)
    parameters = l.iterate()

print (parameters)
