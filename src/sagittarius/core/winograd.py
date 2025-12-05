#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 winograd.py
# dir  := .
# kid  :=

import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce

def default_nodes(n):
    """generate n nodes: [inf, 1, -1, 2, -2, ... , 0]"""
    nodes = [np.inf]
    k = 1
    while len(nodes) < n-1:
        nodes.append(float(k))
        if len(nodes) >= n-1: break
        nodes.append(float(-k))
        k += 1
    nodes.append(0.0)
    return nodes[:n]

def kernel_eval_matrix(nodes, r):
    """
    G's canonical Vandermonde: evaluate the reversed kernel G*(x) = sum g_t x^{r-1-t}
    rows = nodes; columns t = 0..r-1: T[i, t] = a_i^(r-1-t) (if a_i == inf, the row is [1, 0, ..., 0])
    rhape (n, r), where n = m + r - 1
    """
    n = len(nodes)
    T = np.zeros((n, r), dtype=float)
    deg = r - 1
    for i, a in enumerate(nodes):
        if a == np.inf:
            T[i, 0] = 1.0
        else:
            for t in range(r):
                T[i, t] = (a ** (deg - t))
    return T

def alphas_from_nodes(nodes, m, r, last_zero_sign=-1.0):
    """
    Construct the rows alpha_j (j = 0..m-1) of A^T to satisfy pointwise overlap consistency:
        diag(u^(t)) alpha_{j+1} = diag(u^(t+1)) alpha_j
    Closed forms for each node a:
        a == inf: alpha_0 = 1, others 0
        a == 0:   alpha_{m-1} = last_zero_sign, others 0
        otherwise: alpha_j = a^{-j}
    Return A^T (m*n) and the kernel side Vandermonde T (n*r)
    """
    T = kernel_eval_matrix(nodes, r)
    n = len(nodes)
    AT = np.zeros((m, n), dtype=float)
    for i, a in enumerate(nodes):
        if a == np.inf:
            AT[0, i] = 1.0
        elif a == 0.0:
            AT[m-1, i] = float(last_zero_sign)
        else:
            for j in range(m):
                AT[j, i] = (a ** (-j))
    return AT, T

def assemble_C(nodes, m, r, AT):
    """
    Assemble C = B^{-1} (n*n). Column k corresponds to k = t + j.
    Using closed forms:
        for finite nonzero a, the row entries are a^{r-1-k};
        for the inf row only k = 0 is 1; for the 0 row only k = n-1 is alpha_{m-1}[i].
    """
    n = len(nodes)
    C = np.zeros((n, n), dtype=float)
    for i, a in enumerate(nodes):
        if a == np.inf:
            C[i, 0] = 1.0
        elif a == 0.0:
            C[i, n-1] = AT[m-1, i]
        else:
            for k in range(n):
                C[i, k] = a ** (r-1 - k)
    return C

def invert_transpose(C):
    """B^T = (C^T)^{-1}"""
    return np.linalg.inv(C.T)

def optional_row_scaling(BT, G, max_den=16):
    """
    Apply a diagonal row scaling D to make the rows of B^T as integral / low denominator as possible;
    adjust G simultaneously by D^{-1}
    """
    n = BT.shape[0]
    D = np.eye(n)
    for i in range(n):
        fracs = [Fraction(x).limit_denominator(max_den) for x in BT[i]]
        dens = [f.denominator for f in fracs if f.numerator != 0]
        if dens:
            lcm = lambda a,b: a*b // gcd(a,b)
            s = reduce(lcm, dens, 1)
        else:
            s = 1
        D[i,i] = float(s)
    BT_new = D @ BT
    G_new  = np.linalg.inv(D) @ G
    return BT_new, G_new, D

def derive_winograd_1d(m, r, nodes=None, scale_rows=True, last_zero_sign=-1.0, max_den=16):
    """
    main entry point:
    args: F(m, r)
    return: (A^T, B^T, G, nodes).
    """
    n = m + r - 1
    nodes = default_nodes(n) if nodes is None else nodes
    AT, G_vander = alphas_from_nodes(nodes, m, r, last_zero_sign=last_zero_sign)
    C = assemble_C(nodes, m, r, AT)
    BT = invert_transpose(C)
    G = G_vander
    if scale_rows:
        BT, G, _ = optional_row_scaling(BT, G, max_den=max_den)
    return AT, BT, G, nodes

def winograd_2d(Ht, Wt, r, s):
    AT_x, BT_x, G_x, _ = derive_winograd_1d(Ht, r, scale_rows=True, last_zero_sign=-1.0, max_den=8)
    AT_y, BT_y, G_y, _ = derive_winograd_1d(Wt, s, scale_rows=True, last_zero_sign=-1.0, max_den=8)
    AT = np.kron(AT_x, AT_y)
    BT = np.kron(BT_x, BT_y)
    G = np.kron(G_x, G_y)
    return AT, BT, G

if __name__ == "__main__":
    pass
    AT, BT, G = winograd_2d(4, 4, 3, 3)
    print(AT.shape)
    print(BT.shape)
