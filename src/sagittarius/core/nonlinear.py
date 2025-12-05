from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Chebyshev

@dataclass
class NonlinearReady:
    coeffs: np.ndarray
    a: np.float64
    b: np.float64
    degree: int


@dataclass
class Nonlinear:
    def cheb_coeffs(self, func, degree, a, b, n=4096):
        xs = np.linspace(a, b, n)
        ys = func(xs)
        cheb = Chebyshev.fit(xs, ys, deg=degree, domain=[a, b])
        return cheb.coef.tolist()

    def eval_cheb_series_numpy(self, coeffs, x, a, b):
        cheb = Chebyshev(coeffs, domain=[a, b])
        return cheb(x)

    def to_openfhe_cheb_coeffs(self, coeffs):
        coeffs = list(coeffs)
        if len(coeffs) > 0:
            coeffs[0] *= 2.0
        return np.array(coeffs)

    def __init__(self, func_orig, degree, a=-1.0, b=1.0, in_scale=1.0, in_offset=0.0, out_scale=1.0, out_offset=0.0, n=65536):
        self.degree = degree
        self.n = n
        self.a = a
        self.b = b

        def func(x):
            x = (x - in_offset) / in_scale
            x = func_orig(x)
            x = x * out_scale + out_offset
            return x

        coeffs = self.cheb_coeffs(func, self.degree, self.a, self.b, self.n)
        self.coeffs = self.to_openfhe_cheb_coeffs(coeffs)

    def pack(self, N):
        return NonlinearReady(
            coeffs=self.coeffs,
            a=np.float64(self.a),
            b=np.float64(self.b),
            degree=self.degree,
        )


def pack_nonlinear(func_orig, degree, a=-1.0, b=1.0, in_scale=1.0, in_offset=0.0, out_scale=1.0, out_offset=0.0, n=4096):
    return Nonlinear(func_orig, degree, a, b, in_scale, in_offset, out_scale, out_offset, n)
