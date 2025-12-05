#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 matmul.py
# dir  := /home/jyh/project/winograd
# kid  :=

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, cast

import einops
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from tqdm import tqdm

def convert_from_bchw(x, batchsize, C, H, W, H_real, W_real, tilesize, gap):
    c = Cipher(
        x.flatten(),
        order=Order(batchsize=(math.prod(batchsize), 1, 1), C=C, H=H, W=W, H_real=H_real, W_real=W_real),
    )
    c = c.repack(batchsize=batchsize, tilesize=tilesize, gap=gap)
    return c.v.flatten()


def convert_to_bchw(x, batchsize, C, H, W, H_real, W_real, tilesize, gap):
    c = Cipher(
        x.flatten(),
        order=Order(
            batchsize=batchsize, C=C, H=H, W=W, H_real=H_real, W_real=W_real, tilesize=tilesize, gap=gap
        ),
    )
    c = c.repack(batchsize=(math.prod(batchsize), 1, 1), tilesize=(1, 1), gap=(1, 1))
    return c.v.flatten()

def preprocess(arr, *, pad, **kw):
    arr = np.pad(arr, pad)
    return convert_from_bchw(arr, **kw)

def pad_to_kN(arr, N):
    assert arr.ndim == 1
    n = len(arr)
    k = math.ceil(n / N)
    if n >= k*N:
        return arr
    return np.pad(arr, (0, k*N - len(arr)), constant_values=0)


def plot_diagonals(mat, n_slots):
    rows, cols = mat.tocoo().row, mat.tocoo().col
    T_rows, T_cols = math.ceil(mat.shape[0] / n_slots), math.ceil(mat.shape[1] / n_slots)
    H, W = T_rows * n_slots, T_cols * n_slots

    _, ax = plt.subplots()

    h_lines = [([0, W], [i * n_slots, i * n_slots]) for i in range(T_rows + 1)]
    v_lines = [([i * n_slots, i * n_slots], [0, H]) for i in range(T_cols + 1)]
    segs = [[(x1, y1), (x2, y2)] for (xs, ys) in h_lines + v_lines for (x1, x2), (y1, y2) in [(xs, ys)]]
    ax.add_collection(LineCollection(segs, colors="black", linewidths=1, antialiased=False))

    ax.scatter(cols, rows, s=10, c="red", marker=".", linewidths=0, rasterized=True)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


# credit: orion nyu
def diagonalize_old(
    matrix: sp.csr_matrix,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
):
    """
    For each (slots, slots) block of the input matrix, this function
    extracts the generalized diagonals and stores them in a dictionary.
    Each key ((i,j)) in the dictionary block_{i,j}, and the value is
    another dictionary mapping diagonal indices to their values.

    Args:
        matrix (scipy.sparse.csr_matrix): A 4D tensor representing a weight matrix
            for a fully-connected or convolutional layer. The shape must
            conform to (num_blocks_y, num_blocks_x, slots, slots).
        slots (int): The number of SIMD plaintext slots, dictating the
            block size.

    Returns:
        dict: A dictionary where each key is a tuple (i, j) corresponding
              to the (i, j)th (slots, slots) block of `matrix`. The value
              for each key is another dictionary that maps diagonal indices
              within the block to the diagonal's tensor values.

    Examples:
        >>> matrix = torch.tensor([[[[ 0,  1,  2,  3],
                                     [ 4,  5,  6,  7],
                                     [ 8,  9, 10, 11],
                                     [12, 13, 14, 15]]]])
        >>> # Example with slots=4, showing processing of a single block
        >>> print(diagonalize(matrix, slots=4))
        {(0, 0): {0: [0., 5., 10., 15.],
                  1: [1., 6., 11., 12.],
                  2: [2., 7., 8., 13.],
                  3: [3., 4., 9., 14.]}}

        >>> # Example with slots=2, showing processing of four blocks or
              sub-matrices
        >>> print(diagonalize(matrix, slots=2))
        {(0, 0): {0: [0., 5.],
                  1: [1., 4.]},
         (0, 1): {0: [2., 7.],
                  1: [3., 6.]},
         (1, 0): {0: [8., 13.],
                  1: [9., 12.]},
         (1, 1): {0: [10., 15.],
                  1: [11., 14.]}}
    """

    matrix_height, matrix_width = matrix.shape
    num_block_rows = math.ceil(matrix_height / num_slots)
    num_block_cols = math.ceil(matrix_width / num_slots)
    print(f"├── embed method: {embed_method}")
    print(f"├── original matrix shape: {matrix.shape}")
    print(f"├── # blocks (rows, cols) = {(num_block_rows, num_block_cols)}")

    if num_block_rows == 1 and embed_method == "hybrid" and not is_last_layer:
        block_height = 2 ** math.ceil(math.log2(matrix_height))
        output_rotations = int(math.log2(num_slots // block_height))
    else:
        block_height = num_slots
        output_rotations = 0

    # Inflate dimensions of the sparse matrix
    matrix.resize(num_block_rows * block_height, num_block_cols * num_slots)

    print(f"├── resized matrix shape: {matrix.shape}")
    print(f"├── # output rotations: {output_rotations}")

    # Prepare indices for diagonal extraction
    row_idx = torch.arange(block_height).repeat(num_slots // block_height)
    col_idx = torch.arange(block_height)[:, None] + torch.arange(num_slots)[None, :]
    col_idx = torch.where(col_idx >= num_slots, col_idx - num_slots, col_idx)

    diagonals_by_block = {}
    total_diagonals = 0

    # Process each block
    progress_bar = tqdm(
        total=num_block_rows * num_block_cols,
        desc="|    Processing blocks",
        leave=False,
    )
    start_time = time.time()
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            row_start = num_slots * block_row
            col_start = num_slots * block_col
            block_sparse = matrix[
                row_start : row_start + block_height,
                col_start : col_start + num_slots,
            ]
            block_dense = torch.tensor(block_sparse.todense(), dtype=torch.float64)
            block_diagonals = block_dense[row_idx, col_idx]

            # Collect non-zero diagonals
            nonzero_diagonals = {}
            for i in range(block_height):
                if torch.any(block_diagonals[i]):
                    # if torch.sum(torch.abs(block_diagonals[i])) > eps:
                    nonzero_diagonals[i] = block_diagonals[i].tolist()

            total_diagonals += len(nonzero_diagonals)
            diagonals_by_block[(block_row, block_col)] = (
                # nonzero_diagonals or {0: [0.0] * num_slots}
                nonzero_diagonals
                or {}
            )

            progress_bar.set_postfix(
                {
                    "Current Block": f"({block_row},{block_col})",
                    "Total Diagonals": total_diagonals,
                }
            )
            progress_bar.update(1)

    progress_bar.close()
    elapsed_time = time.time() - start_time
    print(f"├── time to pack (s): {elapsed_time:.2f}")
    print(f"├── # diagonals = {total_diagonals}")

    return diagonals_by_block, num_block_rows, num_block_cols, output_rotations


# adapted from orion: https://github.com/baahl-nyu/orion/blob/main/orion/core/packing.py#L182
def diagonalize(
    matrix: sp.csr_matrix,
    num_slots: int,
    verbose: bool = True,
):
    matrix_height, matrix_width = cast(Tuple[int, int], matrix.shape)

    num_block_rows = math.ceil(matrix_height / num_slots)
    num_block_cols = math.ceil(matrix_width / num_slots)
    if verbose:
        print(f"├── original matrix shape: {matrix.shape}")
        print(f"├── # blocks (rows, cols) = {(num_block_rows, num_block_cols)}")

    if num_block_rows == 1:
        block_height = 2 ** math.ceil(math.log2(matrix_height))
        output_rotations = int(math.log2(num_slots // block_height))
    else:
        block_height = num_slots
        output_rotations = 0

    if num_slots % block_height != 0:
        raise ValueError(f"num_slots ({num_slots}) not dividable by block_height ({block_height})。")

    logical_height = num_block_rows * block_height
    logical_width = num_block_cols * num_slots
    if verbose:
        print(f"├── resized matrix shape (logical): {(logical_height, logical_width)}")
        print(f"├── # output rotations: {output_rotations}")

    t0 = time.time()

    coo = matrix.tocoo(copy=False)
    r: NDArray[np.int64] = coo.row.astype(np.int64, copy=False)
    c: NDArray[np.int64] = coo.col.astype(np.int64, copy=False)
    v: NDArray[np.float64] = coo.data.astype(np.float64, copy=False)

    diagonals_nd: Dict[Tuple[int, int], Dict[int, NDArray[np.float64]]] = {}

    if v.size > 0:
        br = r // num_slots
        bc = c // num_slots
        lr = r - br * num_slots
        lc = c - bc * num_slots

        diag_id = (lc - lr) % block_height
        j_index = (lc - diag_id) % num_slots
        iterator = tqdm(
            range(v.size),
            total=v.size,
            desc="Packing diagonals (nnz)",
            unit="nz",
            smoothing=0.1,
            miniters=max(1, v.size // 2000),
            disable=not verbose
        )

        for k in iterator:
            key = (int(br[k]), int(bc[k]))
            di = int(diag_id[k])
            jj = int(j_index[k])
            val = float(v[k])

            inner = diagonals_nd.get(key)
            if inner is None:
                inner = {}
                diagonals_nd[key] = inner

            row_vec = inner.get(di)
            if row_vec is None:
                row_vec = np.zeros(num_slots, dtype=np.float64)
                inner[di] = row_vec

            row_vec[jj] = val

    diagonals_by_block: Dict[Tuple[int, int], Dict[int, List[float]]] = {}
    total_diagonals = 0
    for blk_key, inner in diagonals_nd.items():
        out_inner: Dict[int, List[float]] = {}
        for di, arr in inner.items():
            out_inner[di] = arr.tolist()
        diagonals_by_block[blk_key] = out_inner
        total_diagonals += len(out_inner)

    for br_i in range(num_block_rows):
        for bc_j in range(num_block_cols):
            diagonals_by_block.setdefault((br_i, bc_j), {})

    if verbose:
        print(f"├── time to pack (s): {time.time() - t0:.2f}")
        print(f"├── # diagonals = {total_diagonals}")

    return diagonals_by_block, num_block_rows, num_block_cols, output_rotations


def num_diags(mat, n_slots):
    H, W = mat.shape
    # assert H % n_slots == 0 and W % n_slots == 0
    Br, Bc = math.ceil(H / n_slots), math.ceil(W / n_slots)

    coo = mat.tocoo()
    mask = coo.data != 0
    r = coo.row[mask]
    c = coo.col[mask]

    br = r // n_slots
    bc = c // n_slots
    rl = r % n_slots
    cl = c % n_slots
    d = (cl - rl) % n_slots

    present = np.zeros((Br, Bc, n_slots), dtype=bool)
    present[br, bc, d] = True

    counts = present.sum(axis=2).astype(np.int32)
    return counts


def get_packing_map(H, W, depth):

    def dfs(H, W, start, k):
        if k >= depth:
            mat = np.zeros((H, W))
            for i in range(H):
                for j in range(W):
                    mat[i, j] = (4**k) * (i * W + j) + start
            return mat
        m0 = dfs(H // 2, W // 2, start + 0 * (4**k), k + 1)
        m1 = dfs(H // 2, W // 2, start + 1 * (4**k), k + 1)
        m2 = dfs(H // 2, W // 2, start + 2 * (4**k), k + 1)
        m3 = dfs(H // 2, W // 2, start + 3 * (4**k), k + 1)
        mat = np.block([[m0, m1], [m2, m3]])
        return mat

    mat = dfs(H, W, 0, 0).flatten().tolist()
    p2v = np.asarray(mat, dtype=np.int32)
    v2p = np.empty_like(p2v)
    v2p[p2v] = np.arange(H * W, dtype=np.int32)
    return p2v, v2p


@dataclass
class Order:
    """
    (
        tilesize[0],
        tilesize[1],
        B[0],
        C // (gap[0] * gap[1]),
        B[1],
        H // tilesize[0],
        gap[0],
        W // tilesize[1],
        gap[1],
        B[2]
    ).flatten()
    """

    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        C_real: int | None = None,
        H_real: int | None = None,
        W_real: int | None = None,
        batchsize: tuple[int, int, int] = (1, 1, 1),
        tilesize: tuple[int, int] = (1, 1),
        gap: tuple[int, int] = (1, 1),
    ):
        self.batchsize = batchsize
        self.H = H
        self.W = W
        self.H_real = H_real if H_real is not None else H
        self.W_real = W_real if W_real is not None else W
        self.tilesize = tilesize
        self.C = C
        self.C_real = C_real if C_real is not None else C
        self.gap = gap

        x = np.arange(math.prod(batchsize) * C * H * W).reshape(
            tilesize[0],
            tilesize[1],
            batchsize[0],
            C // (gap[0] * gap[1]),
            batchsize[1],
            H // tilesize[0],
            gap[0],
            W // tilesize[1],
            gap[1],
            batchsize[2],
        )
        x = einops.rearrange(
            x,
            "ht wt bo cg bi hi g1 wi g2 bI -> (bo bi bI) (cg g1 g2) (hi ht) (wi wt)",
            ht=tilesize[0],
            wt=tilesize[1],
            hi=H // tilesize[0],
            wi=W // tilesize[1],
        )
        self.perm = x.ravel().flatten()

    def __str__(self):
        return f"Order(C={self.C}, H={self.H}, W={self.W}, Ht={self.Ht}, Wt={self.Wt}, gap={self.gap})"

    @property
    def B(self):
        return math.prod(self.batchsize)

    @property
    def Ht(self):
        return self.tilesize[0]

    @property
    def Wt(self):
        return self.tilesize[1]

    @property
    def NHt(self):
        return self.H // self.Ht

    @property
    def NWt(self):
        return self.W // self.Wt

    def INDEX(self, i, shape):
        assert len(i) == len(shape)
        result, base = 0, 1
        for idx, s in zip(reversed(i), reversed(shape)):
            result += idx * base
            base *= s
        return result

    def idx(self, bi, ci, hi, wi):
        idx = bi * self.C * self.H * self.W + ci * self.H * self.W + hi * self.W + wi
        return int(self.perm[idx])

    def idx_batch(self, bis, cis, his, wis) -> np.ndarray:
        b = np.asarray(bis)
        c = np.asarray(cis)
        h = np.asarray(his)
        w = np.asarray(wis)
        bb, cc, hh, ww = np.broadcast_arrays(b, c, h, w)

        idx = ((bb * self.C + cc) * self.H + hh) * self.W + ww
        idx = idx.astype(np.int64, copy=False)

        return np.asarray(self.perm, dtype=np.int64)[idx]


class Cipher:
    def __init__(self, v, order):
        self.v = v.flatten()
        self.order = order

    @property
    def H(self):
        return self.order.H

    @property
    def W(self):
        return self.order.W

    @property
    def H_real(self):
        return self.order.H_real

    @property
    def W_real(self):
        return self.order.W_real

    @property
    def Ht(self):
        return self.order.Ht

    @property
    def Wt(self):
        return self.order.Wt

    @property
    def NHt(self):
        return self.order.NHt

    @property
    def NWt(self):
        return self.order.NWt

    @property
    def batchsize(self):
        return self.order.batchsize

    @property
    def B(self):
        return self.order.B

    @property
    def C(self):
        return self.order.C

    @property
    def C_real(self):
        return self.order.C_real

    @property
    def gap(self):
        return self.order.gap

    def repack(self, batchsize, tilesize, gap):
        assert self.B == math.prod(batchsize)
        assert self.H % tilesize[0] == 0 and self.W % tilesize[1] == 0
        assert self.C % math.prod(gap) == 0

        order = Order(
            batchsize=batchsize,
            C=self.C,
            H=self.H,
            W=self.W,
            C_real=self.C_real,
            H_real=self.H_real,
            W_real=self.W_real,
            tilesize=tilesize,
            gap=gap,
        )
        x = self.v.reshape(
            self.Ht,
            self.Wt,
            self.batchsize[0],
            self.C // math.prod(gap),
            self.batchsize[1],
            self.NHt,
            self.gap[0],
            self.NWt,
            self.gap[1],
            self.batchsize[2],
        )
        x = einops.rearrange(
            x,
            "ht wt bo cg bi hi go wi gi bI -> (bo bi bI) (cg go gi) (hi ht) (wi wt)",
        )
        x = einops.rearrange(
            x,
            "(bo bi bI) (cg go gi) (hi ht) (wi wt) -> ht wt bo cg bi hi go wi gi bI",
            cg=self.C // math.prod(gap),
            go=gap[0],
            gi=gap[1],
            ht=tilesize[0],
            wt=tilesize[1],
            hi=self.H // tilesize[0],
            wi=self.W // tilesize[1],
            bo=batchsize[0],
            bi=batchsize[1],
            bI=batchsize[2],
        )
        return Cipher(x.ravel(), order)


@dataclass
class Diagonal:
    bs: int
    gs: int
    data: np.ndarray
    # ptxt: bytes = b""


@dataclass
class Block:
    bx: int
    by: int
    diags: list[Diagonal]


def find_best_n1(diags, N, ratio):
    s = set()
    for _, block in diags.items():
        for i in block.keys():
            s.add(i)
    s = list(s)

    def check(n1):
        bs = set()
        gs = set()
        for i in s:
            bs.add(i % n1)
            gs.add((i // n1) * n1)
        bs = [i for i in bs if i != 0]
        gs = [i for i in gs if i != 0]
        return len(bs), len(gs)

    min_sum = 1e9
    min_n1 = -1
    for n1 in range(1, N + 1):
        n_bs, n_gs = check(n1)
        if n_bs + (n_gs * ratio) < min_sum:
            min_sum = n_bs + (n_gs * ratio)
            min_n1 = n1
    print(min_n1, min_sum)
    return min_n1


class Layer(Enum):
    LINEARTRANSFORM = auto()
    WINOGRAD = auto()
    HERPN = auto()
    NONLINEAR = auto()


@dataclass
class LinearTransformReady:
    T_rows: int
    T_cols: int
    output_rotations: int
    mat: list[list[Block]]
    bias: np.ndarray
    v_bs: list[list[int]]
    num_diags: int
    global_rots: set[int]


@dataclass
class LinearTransform:
    mat: sp.csr_matrix
    bias: np.ndarray | None = None

    def __init__(self, mat, bias=None):
        self.mat = mat
        self.bias = bias

    def pack(self, N):
        diags, T_rows, T_cols, output_rotations = diagonalize(self.mat, N)
        num_diags = sum(len(block) for _, block in diags.items())
        print(f"linear transform: num_digas={num_diags}")

        global_rots = set()
        n1 = find_best_n1(diags, N, 1)
        for _, block in diags.items():
            for d in block.keys():
                bs = d % n1
                gs = (d // n1) * n1
                if bs != 0:
                    global_rots.add(bs % N)
                if gs != 0:
                    global_rots.add(gs % N)

        blocks = [[] for _ in range(T_rows)]
        for bx, by in sorted(diags.keys()):
            ds = diags[(bx, by)]
            block = Block(bx=bx, by=by, diags=[])
            for off in sorted(ds.keys()):
                data = ds[off]
                bs = off % n1
                gs = (off // n1) * n1
                diag = Diagonal(bs=bs % N, gs=gs % N, data=np.roll(data, gs))
                block.diags.append(diag)
            blocks[bx].append(block)

        v_bs = []
        for by in range(T_cols):
            rs = set()
            for bx in range(T_rows):
                for diag in blocks[bx][by].diags:
                    rs.add(diag.bs)
            v_bs.append(sorted(rs))

        return LinearTransformReady(
            T_rows=T_rows,
            T_cols=T_cols,
            output_rotations=output_rotations,
            mat=blocks,
            bias=pad_to_kN(self.bias, N).reshape(-1, N) if self.bias is not None else np.zeros((T_rows, N)),
            v_bs=v_bs,
            num_diags=num_diags,
            global_rots=global_rots,
        )
