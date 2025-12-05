#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := time python3 test.py
# dir  := .
# kid  :=

from sagittarius.core import Order, Cipher
from sagittarius.core import pack_conv2d_toeplitz, pack_conv2d_winograd, Batchnorm2d
import numpy as np
import random
import torch

def order_idx_test():
    Ht = random.randint(1, 3)
    Wt = random.randint(1, 3)
    gap = random.randint(1, 4)
    depth = random.randint(0, 2)
    H = random.randint(1, 10) * Ht * (1 << depth)
    W = random.randint(1, 10) * Wt * (1 << depth)
    C = random.randint(1, 10) * (gap * gap)
    print(f"H={H}, W={W}, Ht={Ht}, Wt={Wt}, gap={gap}, depth={depth}")

    order = Order(C=C, H=H, W=W, Ht=Ht, Wt=Wt, gap=gap, depth=depth)

    img = np.arange(C*H*W).reshape(Ht, Wt, C//(gap*gap), H*gap//Ht, W*gap//Wt)
    out = np.zeros((C, H, W))

    for hti in range(Ht):
        for wti in range(Wt):
            for ci in range(C//(gap*gap)):
                for hi in range(H//Ht):
                    for wi in range(W//Wt):
                        for gi in range(gap):
                            for gj in range(gap):
                                idx = order.v2p[hi*(W//Wt) + wi]
                                hip, wip = idx // (W//Wt), idx % (W//Wt)
                                m = ci*gap*gap + gi*gap + gj
                                h = hti + hip*Ht
                                w = wti + wip*Wt
                                out[m, h, w] = img[hti, wti, ci, hi*gap + gi, wi*gap + gj]

    cis, his, wis = [], [], []
    ans = []
    for ci in range(C):
        for hi in range(H):
            for wi in range(W):
                cis.append(ci)
                his.append(hi)
                wis.append(wi)
                i = order.idx(ci, hi, wi)
                j = int(out[ci, hi, wi])
                ans.append(j)
                assert i == j
    out = order.idx_batch(cis, his, wis, as_numpy=True)
    assert np.array_equal(out, np.array(ans))

def cipher_repack_test():
    Ht = random.randint(1, 3)
    Wt = random.randint(1, 3)
    gap = random.randint(1, 4)
    depth = random.randint(0, 2)
    H = random.randint(1, 10) * Ht * (1 << depth)
    W = random.randint(1, 10) * Wt * (1 << depth)
    C = random.randint(1, 10) * (gap * gap)
    print(f"H={H}, W={W}, Ht={Ht}, Wt={Wt}, gap={gap}, depth={depth}")
    order = Order(C=C, H=H, W=W)
    img = np.arange(C*H*W)
    cipher = Cipher(img, order)
    cipher = cipher.repack(Ht=Ht, Wt=Wt, gap=gap, depth=depth)
    for ci in range(C):
        for hi in range(H):
            for wi in range(W):
                i = cipher.order.idx(ci, hi, wi)
                assert cipher.v[i] == ci*H*W + hi*W + wi

def conv2d_toeplitz_test():
    stride = random.randint(1, 2)
    Ht = random.randint(1, 2)
    Wt = random.randint(1, 2)
    gap = random.randint(1, 4)
    depth = random.randint(0, 2)
    H_real = random.randint(3, 6) * Ht * (1 << depth) * stride
    W_real = random.randint(3, 6) * Wt * (1 << depth) * stride
    H = H_real + (random.randint(0, 2) * Ht * (1 << depth) * stride)
    W = W_real + (random.randint(0, 2) * Wt * (1 << depth) * stride)
    C = random.randint(1, 6) * (gap * gap)
    M = random.randint(1, 6) * (gap * gap)
    R = S = random.choice([3, 5, 7])
    print(f"H={H}, W={W}, Ht={Ht}, Wt={Wt}, gap={gap}, depth={depth}, H_real={H_real}, W_real={W_real}, stride={stride}, C={C}, M={M}, R={R}, S={S}")

    input_raw = np.random.rand(C, H_real, W_real)
    input = np.pad(input_raw, ((0, 0), (0, H-H_real), (0, W-W_real)))

    bn_mean = np.random.rand(M)
    bn_var = np.random.rand(M)
    gamma = np.random.rand(M)
    beta = np.random.rand(M)
    eps = 1e-5

    kernel = np.random.rand(M, C, R, S)

    def ref():
        o = torch.nn.functional.conv2d(torch.tensor(input_raw.reshape(1, C, H_real, W_real)), torch.tensor(kernel), padding=R//2, stride=stride)
        o = torch.nn.functional.batch_norm(o, torch.tensor(bn_mean), torch.tensor(bn_var), torch.tensor(gamma), torch.tensor(beta), eps=eps).numpy().reshape(M, H_real//stride, W_real//stride)
        o = np.pad(o, ((0, 0), (0, (H-H_real)//stride), (0, (W-W_real)//stride)))
        return o
    conv_ref = ref()

    lt = pack_conv2d_toeplitz(
        Order(C=C, H=H, W=W, H_real=H_real, W_real=W_real),
        Order(C=M, H=H//stride, W=W//stride, H_real=H_real//stride, W_real=W_real//stride, Ht=Ht, Wt=Wt, gap=gap),
        stride=stride,
        kernel=kernel,
        batchnorm=Batchnorm2d(mean=bn_mean, var=bn_var, gamma=gamma, beta=beta, eps=eps)
    )

    output = lt.mat @ input.flatten()
    if lt.bias is not None:
        output = output + lt.bias.flatten()
    out = Cipher(
        output,
        Order(C=M, H=H//stride, W=W//stride, H_real=H_real//stride, W_real=W_real//stride, Ht=Ht, Wt=Wt, gap=gap),
    ).flatten()
    assert np.allclose(conv_ref, out)

def conv2d_winograd2233_test():
    Ht = Wt = 2
    gap = random.randint(1, 4)
    H_real = random.randint(3, 10) * Ht
    W_real = random.randint(3, 10) * Wt
    H = H_real + (random.randint(1, 2) * Ht)
    W = W_real + (random.randint(1, 2) * Wt)
    M = C = random.randint(1, 6) * (gap * gap)
    R = S = 3
    print(f"H={H}, W={W}, Ht={Ht}, Wt={Wt}, gap={gap}, H_real={H_real}, W_real={W_real}, C={C}, M={M}")
    NHt, NWt = H // Ht, W // Wt

    order = Order(C=C, H=H, W=W, H_real=H_real, W_real=W_real, Ht=Ht, Wt=Wt, gap=gap)

    def inner01(weights, operands):
        weights = np.array(weights, dtype=np.int8)
        assert np.isin(weights, (-1, 0, 1)).all()
        idx = np.flatnonzero(weights)
        pairs = list(zip(idx, weights[idx]))
        i, w = pairs.pop(0)
        sum = operands[i] if w == 1 else -operands[i]
        while pairs:
            i, w = pairs.pop(0)
            sum = sum + operands[i] if w == 1 else sum - operands[i]
        return sum

    bn_mean = np.random.rand(M).astype(np.float32)
    bn_var = np.random.rand(M).astype(np.float32)
    gamma = np.random.rand(M).astype(np.float32)
    beta = np.random.rand(M).astype(np.float32)
    eps = 1e-5

    kernel = np.random.rand(M, C, 3, 3)
    wino = pack_conv2d_winograd(order, kernel, batchnorm=Batchnorm2d(mean=bn_mean, var=bn_var, gamma=gamma, beta=beta, eps=eps))

    input_raw = np.random.rand(C, H_real, W_real)
    input_padded = np.pad(input_raw, ((0, 0), (0, H-H_real), (0, W-W_real)))
    cipher = Cipher(input_padded.flatten(), Order(C=C, H=H, W=W))
    cipher = cipher.repack(Ht=Ht, Wt=Wt, gap=gap, depth=0)
    input = cipher.v.reshape(Ht*Wt, -1)

    def toeplitz_ref():
        i = torch.tensor(input_raw.reshape(1, C, H_real, W_real), dtype=torch.float32)
        k = torch.tensor(kernel, dtype=torch.float32)
        o = torch.nn.functional.conv2d(i, k, padding=1)
        o = torch.nn.functional.batch_norm(o, torch.tensor(bn_mean), torch.tensor(bn_var), torch.tensor(gamma), torch.tensor(beta), eps=eps)
        o = o.numpy().reshape(M, H_real, W_real)
        o = np.pad(o, ((0, 0), (0, H-H_real), (0, W-W_real)))
        c = Cipher(o.flatten(), Order(C=M, H=H, W=W))
        c = c.repack(Ht=Ht, Wt=Wt, gap=gap, depth=0)
        return c.v

    ref = toeplitz_ref()

    def rotate(v, off):
        return np.roll(v, -off)

    c = np.empty(((Ht+R-1)*(Wt+S-1), NHt*NWt*M))
    for (i, (targets, offsets)) in enumerate(zip(wino.i2c_target, wino.i2c_offset)):
        for (tar, off) in zip(targets, offsets):
            c[tar] = rotate(input[i], off)

    D_tilde = np.empty(((Ht+R-1)*(Wt+S-1), NHt*NWt*M))
    for i in range((Ht+R-1)*(Wt+S-1)):
        D_tilde[i] = inner01(wino.B_kron[i], c)

    E = np.empty(((Ht+R-1)*(Wt+S-1), NHt*NWt*M))
    for i in range((Ht+R-1)*(Wt+S-1)):
        E[i] = wino.mats[i] @ D_tilde[i]

    y = np.empty((Ht*Wt, NHt*NWt*M))
    for i in range(Ht*Wt):
        y[i] = inner01(wino.A_kron[i], E)

    y = y.flatten()
    if wino.bias is not None:
        y += wino.bias.flatten()
    assert np.allclose(y.flatten(), ref)

def myt():
    C, H, W = 8, 4, 4
    order = Order(C=C, H=H, W=W, Ht=2, Wt=2, gap=2)
    indices = np.zeros((C, H, W))
    for ci in range(C):
        for hi in range(H):
            for wi in range(W):
                # indices[ci, hi, wi] = order.idx(ci, hi, wi)
                o = order.idx(ci, hi, wi)
                on = order.idx_new(ci, hi, wi)
                assert o == on, (ci, hi, wi, o, on)
    exit(0)

    x = np.arange(C*H*W)
    c = Cipher(x, order)
    c = c.repack(Ht=1, Wt=1, gap=1, depth=0)

if __name__ == "__main__":
    # order_idx_test()
    # cipher_repack_test()
    # conv2d_toeplitz_test()
    # conv2d_winograd2233_test()
    myt()
