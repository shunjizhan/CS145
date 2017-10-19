import math
import numpy


def log(n):
    return math.log(n)


def exp(n):
    return math.exp(n)


class logistic:
    def __init__(self, parameters):
        self.parameters = parameters

    def log_likelihood(self):
        ll = 0.0
        return ll

    def gradients(self):
        gradients = []
        return gradients

    def iterate(self):
        return self.parameters

    def hessian(self):
        n = len(self.parameters)
        hessian = numpy.zeros((n, n))
        return hessian


parameters = [0.25, 0.25, 0.25]
iterations = 2
for i in range(iterations):
    l = logistic(parameters)
    parameters = l.iterate()

print (parameters)
