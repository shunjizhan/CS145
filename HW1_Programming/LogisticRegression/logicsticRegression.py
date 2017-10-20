import math
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import normalize


def log(n):
    return math.log(n)


def exp(n):
    return math.exp(n)


def transpose(M):
    return np.transpose(M)


class logistic:
    def __init__(self, parameters):
        self.parameters = parameters
        self.X = normalize(np.array([
            [60.0, 155.0],
            [64.0, 135.0],
            [73.0, 170.0]
        ]), axis=0)
        self.X = np.insert(self.X, 0, 1, axis=1)
        # print self.X
        self.N = len(self.X)                    # number of samples
        self.M = len(self.X[0])                 # beta length
        assert(self.N == 3)
        assert(self.M == 3)
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
        X, Y, beta = self.X, self.Y, self.beta

        gradients = np.zeros(self.N)
        for i in range(self.N):
            exp_betaT_xi = exp(transpose(beta).dot(X[i]))
            p = exp_betaT_xi / (1 + exp_betaT_xi)
            # print exp_betaT_xi
            gradients += X[i] * (Y[i] - p)

        return gradients

    def hessian(self):
        X, beta = self.X, self.beta
        n = len(self.parameters)
        hessian = np.zeros((n, n))

        for j in range(n):
            for i in range(n):
                # print transpose(beta).dot(X[i])
                exp_betaT_xi = exp(transpose(beta).dot(X[i]))
                A = exp_betaT_xi / (1 + exp_betaT_xi) ** 2
                sum = 0
                for a in range(n):
                    sum += X[a][i] * X[a][j] * A
                hessian[j][i] = -1 * sum
        return hessian

    def iterate(self):
        # log_likelihood = self.log_likelihood()
        gradients = self.gradients()
        hessian = self.hessian()
        hessian_inv = inv(hessian)
        self.parameters = self.parameters - hessian_inv.dot(gradients)

        print (gradients)
        return self.parameters


parameters = [0.25, 0.25, 0.25]
iterations = 2
for i in range(iterations):
    l = logistic(parameters)
    parameters = l.iterate()
    print (parameters)
