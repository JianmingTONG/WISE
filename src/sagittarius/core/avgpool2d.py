import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

from sagittarius.core import LinearTransform


def avgpool2d(
    in_order, out_order, size, stride, padding, out_scale=1.0, out_offset=0.0, in_scale=1.0, in_offset=0.0
):

    def reform(x, C):
        if isinstance(x, np.ndarray):
            if x.ndim != 1 or x.shape[0] != C:
                raise ValueError(f"BatchNorm parameter shape mismatch: expected ({C},), got {x.shape}")
            return x
        else:
            return np.array([x] * C)

    assert in_order.B == out_order.B
    assert in_order.C == out_order.C
    assert (in_order.H_real + 2 * padding - size) // stride + 1 == out_order.H_real
    assert (in_order.W_real + 2 * padding - size) // stride + 1 == out_order.W_real
    # assert (in_order.H + 2 * padding - size) // stride + 1 == out_order.H
    # assert (in_order.W + 2 * padding - size) // stride + 1 == out_order.W

    out_scale = reform(out_scale, out_order.C)
    out_offset = reform(out_offset, out_order.C)
    in_scale = reform(in_scale, in_order.C)
    in_offset = reform(in_offset, in_order.C)

    outs = np.zeros((out_order.B * out_order.C * out_order.H * out_order.W))
    outo = np.zeros((out_order.B * out_order.C * out_order.H * out_order.W))
    ino = np.zeros((in_order.B * in_order.C * in_order.H * in_order.W))
    ins = np.zeros((in_order.B * in_order.C * in_order.H * in_order.W))

    for b in range(in_order.B):
        for c in range(in_order.C):
            for h in range(in_order.H):
                for w in range(in_order.W):
                    idx = in_order.idx(bi=b, ci=c, hi=h, wi=w)
                    ins[idx] = in_scale[c]
                    ino[idx] = in_offset[c]
    for b in range(out_order.B):
        for c in range(out_order.C):
            for h in range(out_order.H):
                for w in range(out_order.W):
                    idx = out_order.idx(bi=b, ci=c, hi=h, wi=w)
                    outs[idx] = out_scale[c]
                    outo[idx] = out_offset[c]

    def window(b, c, h0, w0):
        cols, data = [], []
        for dh in range(size):
            hi = h0 + dh
            if hi >= in_order.H_real or hi < 0:
                continue
            for dw in range(size):
                wi = w0 + dw
                if wi >= in_order.W_real or wi < 0:
                    continue
                cols.append(int(in_order.idx(bi=b, ci=c, hi=hi, wi=wi)))
                data.append(1.0 / (size * size))
        return cols, data

    data, rows, cols = [], [], []

    pbar = tqdm(total=out_order.B * out_order.C_real * out_order.H_real * out_order.W_real, desc="avgpool")

    total_flop = 0
    for b in range(out_order.B):
        for c in range(out_order.C_real):
            for h in range(out_order.H_real):
                for w in range(out_order.W_real):
                    r = out_order.idx(bi=b, ci=c, hi=h, wi=w)
                    cs, ds = window(b, c, h * stride - padding, w * stride - padding)
                    assert len(cs) == len(ds)
                    rows.extend([int(r)] * len(cs))
                    cols.extend(cs)
                    data.extend(ds)
                    pbar.update(1)
                    total_flop += 1

    pbar.close()
    mat = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(
            out_order.B * out_order.C * out_order.H * out_order.W,
            in_order.B * in_order.C * in_order.H * in_order.W,
        ),
    )
    print(f"Total FLOP: {total_flop}")

    bias = -outs * (mat @ (ino / ins)) + outo

    mat = mat.tocsr()
    inv_ins = 1.0 / ins
    m_csc = mat.tocsc()
    m_csc.data *= np.repeat(inv_ins, np.diff(m_csc.indptr))
    mat = m_csc.tocsr()
    mat.data *= np.repeat(outs, np.diff(mat.indptr))

    mat.eliminate_zeros()

    return mat, bias


def pack_avgpool2d(
    in_order, out_order, size, stride, padding, out_scale=1.0, out_offset=0.0, in_scale=1.0, in_offset=0.0
):
    mat, bias = avgpool2d(
        in_order, out_order, size, stride, padding, out_scale, out_offset, in_scale, in_offset
    )
    lt = LinearTransform(mat, bias)
    return lt
