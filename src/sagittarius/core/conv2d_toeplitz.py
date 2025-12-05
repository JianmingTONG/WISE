from sagittarius.core import LinearTransform
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

def conv2d_toeplitz(in_order, out_order, R, S, stride, kernel, bn_a=1.0, bn_b=0.0, out_scale=1.0, out_offset=0.0, in_scale=1.0, in_offset=0.0):

    def reform(x, C):
        if isinstance(x, np.ndarray):
            if x.ndim != 1 or x.shape[0] != C:
                raise ValueError(f"BatchNorm parameter shape mismatch: expected ({C},), got {x.shape}")
            return x
        else:
            return np.array([x] * C)

    bn_a = reform(bn_a, out_order.C)
    bn_b = reform(bn_b, out_order.C)

    out_scale = reform(out_scale, out_order.C)
    out_offset = reform(out_offset, out_order.C)
    in_scale = reform(in_scale, in_order.C)
    in_offset = reform(in_offset, in_order.C)

    outs = np.zeros((out_order.B * out_order.C * out_order.H * out_order.W))
    outo = np.zeros((out_order.B * out_order.C * out_order.H * out_order.W))
    ino = np.zeros((in_order.B * in_order.C * in_order.H * in_order.W))
    ins = np.zeros((in_order.B * in_order.C * in_order.H * in_order.W))
    bns = np.zeros((out_order.B * out_order.C * out_order.H * out_order.W))
    bno = np.zeros((out_order.B * out_order.C * out_order.H * out_order.W))

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
                    if c < out_order.C_real and h < out_order.H_real and w < out_order.W_real:
                        bns[idx] = bn_a[c]
                        bno[idx] = bn_b[c]

    assert in_order.B == out_order.B

    def window(b, m, h, w):
        cols, data = [], []
        flop = 0
        for c in range(in_order.C_real):
            for h_off in range(-(R // 2), R // 2 + 1):
                for w_off in range(-(S // 2), S // 2 + 1):
                    if 0 <= h + h_off < in_order.H_real and 0 <= w + w_off < in_order.W_real:
                        idx = int(in_order.idx(bi=b, ci=c, hi=h + h_off, wi=w + w_off))
                        val = kernel[m, c, h_off + R // 2, w_off + S // 2]
                        flop += 1
                        cols.append(idx)
                        data.append(val)
        return cols, data, flop

    B, C, M, H, W = in_order.B, in_order.C, out_order.C, in_order.H, in_order.W

    data, rows, cols = [], [], []

    pbar = tqdm(total=B * out_order.C_real * out_order.H_real * out_order.W_real, desc="conv2d toeplitz")

    total_flop = 0
    for b in range(B):
        for m in range(out_order.C_real):
            for h in range(out_order.H_real):
                for w in range(out_order.W_real):
                    r = out_order.idx(bi=b, ci=m, hi=h, wi=w)
                    cs, ds, flop = window(b, m, h * stride, w * stride)
                    total_flop += flop
                    assert len(cs) == len(ds)
                    rows.extend([int(r)] * len(cs))
                    cols.extend(cs)
                    data.extend(ds)
                    pbar.update(1)
    print(f"Total FLOP: {total_flop}")

    pbar.close()

    mat = sp.csr_matrix((data, (rows, cols)), shape=(B * M * H // stride * W // stride, B * C * H * W))

    bias = -outs * (mat @ (ino / ins)) * bns + outs * bno + outo

    mat = mat.tocsr()
    inv_ins = 1.0 / ins
    m_csc = mat.tocsc()
    m_csc.data *= np.repeat(inv_ins, np.diff(m_csc.indptr))
    mat = m_csc.tocsr()
    mat.data *= np.repeat(outs, np.diff(mat.indptr))
    mat.data *= np.repeat(bns, np.diff(mat.indptr))

    mat.eliminate_zeros()

    return mat, bias

def pack_conv2d_toeplitz(in_order, out_order, stride, kernel, bn_a=1.0, bn_b=0.0, out_scale=1.0, out_offset=0.0, in_scale=1.0, in_offset=0.0):
    R, S = kernel.shape[2], kernel.shape[3]
    mat, bias = conv2d_toeplitz(in_order, out_order, R, S, stride, kernel, bn_a, bn_b, out_scale, out_offset, in_scale, in_offset)
    lt = LinearTransform(mat, bias)
    return lt

