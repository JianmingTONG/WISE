from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from sagittarius.core import Block, Diagonal, diagonalize, find_best_n1

from .winograd import winograd_2d


@dataclass
class WinogradReady:
    T_rows: int
    T_cols: int
    output_rotations: int
    R: int
    S: int
    Ht: int
    Wt: int
    i2c_target: np.ndarray
    i2c_offset: np.ndarray
    A_kron: np.ndarray
    B_kron: np.ndarray
    mats: list[list[list[Block]]]
    bias: np.ndarray
    v_bs: list[list[int]]
    num_diags: int
    global_rots: set[int]


@dataclass
class Winograd:
    mat: sp.csr_matrix
    G_tilde: np.ndarray

    def __init__(
        self, mat, G_tilde, Ht, Wt, R, S, i2c_target, i2c_offset, A_kron, B_kron, C, out_scale, out_offset
    ):
        self.mat = mat
        self.G_tilde = G_tilde
        self.R = R
        self.S = S
        self.Ht = Ht
        self.Wt = Wt
        self.i2c_target = i2c_target
        self.i2c_offset = i2c_offset
        self.A_kron = A_kron
        self.B_kron = B_kron
        self.C = C
        self.out_scale = out_scale
        self.out_offset = out_offset

    def pack(self, N):
        print(np.min(self.out_scale), np.max(self.out_scale))
        T_rows, T_cols = 0, 0
        diag_base, T_rows, T_cols, output_rotations = diagonalize(self.mat, N)

        num_diags = sum(len(block) for _, block in diag_base.items())
        print(f"winograd: num_digas={num_diags}")

        n1 = find_best_n1(diag_base, N, 1)

        global_rots = set()
        for _, block in diag_base.items():
            for d in block.keys():
                bs = d % n1
                gs = (d // n1) * n1
                if bs != 0:
                    global_rots.add(bs % N)
                if gs != 0:
                    global_rots.add(gs % N)

        mats = []
        G_tilde = self.G_tilde.reshape(-1, (self.Ht + self.R - 1) * (self.Wt + self.S - 1))
        for i in range((self.Ht + self.R - 1) * (self.Wt + self.S - 1)):
            blocks = [[] for _ in range(T_rows)]
            for bx, by in sorted(diag_base.keys()):
                ds = diag_base[(bx, by)]
                block = Block(bx=bx, by=by, diags=[])
                for off in sorted(ds.keys()):
                    data = ds[off]
                    bs = off % n1
                    gs = (off // n1) * n1
                    data_round = [round(k) for k in data]
                    d = [
                        0 if k == 0 else G_tilde[k - 1, i] * self.out_scale[(k - 1) // self.C]
                        for k in data_round
                    ]
                    diag = Diagonal(bs=bs % N, gs=gs % N, data=np.roll(d, gs))
                    block.diags.append(diag)
                blocks[bx].append(block)
            mats.append(blocks)

        v_bs = []
        for by in range(T_cols):
            rs = set()
            for bx in range(T_rows):
                for diag in mats[0][bx][by].diags:
                    rs.add(diag.bs)
            v_bs.append(sorted(rs))

        return WinogradReady(
            T_rows=T_rows,
            T_cols=T_cols,
            output_rotations=output_rotations,
            R=self.R,
            S=self.S,
            Ht=self.Ht,
            Wt=self.Wt,
            i2c_target=self.i2c_target,
            i2c_offset=self.i2c_offset,
            A_kron=self.A_kron,
            B_kron=self.B_kron,
            mats=mats,
            bias=(
                self.out_offset.reshape(-1, N)
                if self.out_offset is not None
                else np.zeros((T_rows * self.Ht * self.Wt, N))
            ),
            v_bs=v_bs,
            num_diags=num_diags,
            global_rots=global_rots,
        )


def conv2d_winograd(
    in_order,
    out_order,
    kernel,
    bn_a=1.0,
    bn_b=0.0,
    out_scale=1.0,
    out_offset=0.0,
    in_scale=1.0,
    in_offset=0.0,
):

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

    R, S = kernel.shape[2], kernel.shape[3]
    assert R == 3 and S == 3
    assert in_order.Ht == 2 and in_order.Wt == 2
    assert out_order.Ht == 2 and out_order.Wt == 2
    assert in_order.H_real == out_order.H_real
    assert in_order.W_real == out_order.W_real
    assert in_order.Ht == out_order.Ht
    assert in_order.Wt == out_order.Wt
    assert in_order.H == out_order.H
    assert in_order.W == out_order.W
    assert in_order.B == out_order.B
    _, _, G_kron = winograd_2d(in_order.Ht, in_order.Wt, R, S)

    B = in_order.B
    Ht, Wt = in_order.Ht, in_order.Wt

    def window(b, m, h, w):
        cols = []
        data = [(m, c) for c in range(in_order.C_real)]
        for c in range(in_order.C_real):
            cols.append(int(in_order.idx(bi=b, ci=c, hi=h, wi=w)))
        return cols, data

    data, rows, cols = [], [], []

    pbar = tqdm(
        total=B * out_order.C_real * in_order.H_real * in_order.W_real // in_order.Ht // in_order.Wt,
        desc="conv2d winograd",
    )

    for b in range(B):
        for m in range(out_order.C_real):
            for h in range(0, out_order.H_real, Ht):
                for w in range(0, out_order.W_real, Wt):
                    r = out_order.idx(bi=b, ci=m, hi=h, wi=w)
                    cs, ds = window(b, m, h, w)
                    assert len(cs) == len(ds)
                    rows.extend([int(r)] * len(cs))
                    cols.extend(cs)
                    data.extend([i * in_order.C + j + 1 for i, j in ds])
                    pbar.update(1)

    pbar.close()
    G_tilde = np.matmul(G_kron[:, :], kernel.reshape(out_order.C, in_order.C, -1)[..., None]).squeeze(-1)

    mat = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(
            out_order.B * out_order.C * out_order.NHt * out_order.NWt,
            in_order.B * in_order.C * in_order.NHt * in_order.NWt,
        ),
    )

    return mat, G_tilde, out_scale * bn_a, outs * bno + outo


def pack_conv2d_winograd(
    in_order,
    out_order,
    kernel,
    bn_a=1.0,
    bn_b=0.0,
    out_scale=1.0,
    out_offset=0.0,
    in_scale=1.0,
    in_offset=0.0,
):
    R, S = kernel.shape[2], kernel.shape[3]
    Ht, Wt = in_order.Ht, in_order.Wt
    NWt = in_order.W // Wt
    gap_outer, gap_inner = in_order.gap[0], in_order.gap[1]
    if Ht != 2 or Wt != 2 or R != 3 or S != 3:
        raise NotImplementedError("Winograd not implemented yet")

    A_kron, B_kron, _ = winograd_2d(Ht, Wt, R, S)
    input2c_target = np.array([[5, 7, 13, 15], [4, 6, 12, 14], [1, 3, 9, 11], [0, 2, 8, 10]])
    input2c_offset = np.array(
        [
            [0, gap_inner, NWt * gap_outer * gap_inner, NWt * gap_outer * gap_inner + gap_inner],
            [-gap_inner, 0, NWt * gap_outer * gap_inner - gap_inner, NWt * gap_outer * gap_inner],
            [-NWt * gap_outer * gap_inner, -NWt * gap_outer * gap_inner + gap_inner, 0, gap_inner],
            [-NWt * gap_outer * gap_inner - gap_inner, -NWt * gap_outer * gap_inner, -gap_inner, 0],
        ]
    )

    mat, G_tilde, out_scale, out_offset = conv2d_winograd(
        in_order, out_order, kernel, bn_a, bn_b, out_scale, out_offset, in_scale, in_offset
    )

    wino = Winograd(
        mat=mat,
        G_tilde=G_tilde,
        Ht=Ht,
        Wt=Wt,
        R=R,
        S=S,
        i2c_target=input2c_target,
        i2c_offset=input2c_offset,
        A_kron=A_kron,
        B_kron=B_kron,
        C=in_order.C,
        out_scale=out_scale,
        out_offset=out_offset,
    )
    return wino
