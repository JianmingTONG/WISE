from dataclasses import dataclass
import numpy as np
from sagittarius.core import Order, pad_to_kN

@dataclass
class HerPNReady:
    a0: np.ndarray
    a1: np.ndarray

@dataclass
class HerPN:
    order: Order
    a0: np.ndarray
    a1: np.ndarray
    a2: np.ndarray

    def __init__(self, order, a0, a1, a2):
        self.order = order
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    def pack(self, N):
        total = self.order.B * self.order.C * self.order.H * self.order.W
        c_a0 = np.zeros(total)
        c_a1 = np.zeros(total)
        for b in range(self.order.B):
            for c in range(self.order.C_real):
                for h in range(self.order.H_real):
                    for w in range(self.order.W_real):
                        i = self.order.idx(b, c, h, w)
                        c_a0[i] = self.a0[c]
                        c_a1[i] = self.a1[c]
        return HerPNReady(a0=pad_to_kN(c_a0, N).reshape(-1, N), a1=pad_to_kN(c_a1, N).reshape(-1, N))


def pack_herpn(order, a0, a1, a2):
    assert len(a0.shape) == 1 and len(a1.shape) == 1 and len(a2.shape) == 1
    assert a0.shape[0] == order.C_real and a1.shape[0] == order.C_real and a2.shape[0] == order.C_real
    return HerPN(order, a0, a1, a2)

