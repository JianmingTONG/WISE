import numpy as np
from scipy import sparse as sp
from tqdm.auto import tqdm

from sagittarius.core import LinearTransform


def pack_fc(in_order, out_order, weight, bias=None, out_scale=1.0, out_offset=0.0, in_scale=1.0, in_offset=0.0):

    def reform(x, C):
        if isinstance(x, np.ndarray):
            if x.ndim != 1 or x.shape[0] != C:
                raise ValueError(f"BatchNorm parameter shape mismatch: expected ({C},), got {x.shape}")
            return x
        else:
            return np.array([x] * C)

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

    assert in_order.B == out_order.B
    B = in_order.B

    bias = (
        np.zeros((out_order.C, out_order.H, out_order.W)) if bias is None else bias.reshape(-1, 1, 1)
    ).flatten()

    weight = weight.reshape(
        out_order.C_real,
        out_order.H_real,
        out_order.W_real,
        in_order.C_real,
        in_order.H_real,
        in_order.W_real,
    )
    weight = np.pad(
        weight,
        (
            (0, 0),
            (0, out_order.H - out_order.H_real),
            (0, out_order.W - out_order.W_real),
            (0, 0),
            (0, in_order.H - in_order.H_real),
            (0, in_order.W - in_order.W_real),
        ),
        constant_values=0,
    )
    rows, cols, data = [], [], []
    bias_out = np.zeros(B * out_order.C * out_order.H * out_order.W)
    pbar = tqdm(total=B * out_order.C_real * out_order.H_real * out_order.W_real, desc="fully connected")
    total_flop = 0
    for b in range(B):
        for c_out in range(out_order.C):
            for h_out in range(out_order.H):
                for w_out in range(out_order.W):
                    idx_out = out_order.idx(bi=b, ci=c_out, hi=h_out, wi=w_out)

                    if c_out >= out_order.C_real or h_out >= out_order.H_real or w_out >= out_order.W_real:
                        bias_out[idx_out] = out_offset[c_out]
                        continue

                    bias_out[idx_out] = out_scale[c_out] * bias[c_out] + out_offset[c_out]
                    for c_in in range(in_order.C_real):
                        for h_in in range(in_order.H_real):
                            for w_in in range(in_order.W_real):
                                idx_in = in_order.idx(bi=b, ci=c_in, hi=h_in, wi=w_in)
                                val = weight[c_out, h_out, w_out, c_in, h_in, w_in] * out_scale[c_out]
                                rows.append(idx_out)
                                cols.append(idx_in)
                                data.append(val)
                                total_flop += 1
                    pbar.update(1)

    print(f"Total FLOP: {total_flop}")
    pbar.close()

    mat = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(B * out_order.C * out_order.H * out_order.W, B * in_order.C * in_order.H * in_order.W),
    )

    mat.eliminate_zeros()

    return LinearTransform(mat, bias_out)
