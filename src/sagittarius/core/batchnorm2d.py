from dataclasses import dataclass
import numpy as np

@dataclass
class Batchnorm2d:
    mean: np.ndarray
    var: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    eps: float

    def __init__(self, mean, var, gamma, beta, eps):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    @property
    def a(self):
        return self.gamma / np.sqrt(self.var + self.eps)

    @property
    def b(self):
        return self.beta - self.gamma * (self.mean / np.sqrt(self.var + self.eps))

