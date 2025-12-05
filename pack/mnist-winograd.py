#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 mnist-winograd.py
# dir  := .
# kid  :=

import math
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

from sagittarius.core import (
    Layer,
    Order,
    pack_avgpool2d,
    pack_conv2d_toeplitz,
    pack_conv2d_winograd,
    pack_fc,
    pack_herpn,
    preprocess,
)

BASE_DIR = "../data/mnist-winograd"
WEIGHT_PATH = "../clear/mnist/mnist_cnn_herpn_avgpool_wo_padding_196.pth"

Path(f"{BASE_DIR}/raw").mkdir(parents=True, exist_ok=True)
Path(f"{BASE_DIR}/ready").mkdir(parents=True, exist_ok=True)


def herpn_eval(sd, prefix):
    eps = 1e-5
    gamma = sd[prefix + ".gamma"]
    beta = sd[prefix + ".beta"]

    ft0 = sd[prefix + ".f_tilde_0"]
    ft1 = sd[prefix + ".f_tilde_1"]
    ft2 = sd[prefix + ".f_tilde_2"]

    mu0 = sd[prefix + ".running_mean_h0"]
    mu1 = sd[prefix + ".running_mean_h1"]
    mu2 = sd[prefix + ".running_mean_h2"]

    var0 = sd[prefix + ".running_var_h0"]
    var1 = sd[prefix + ".running_var_h1"]
    var2 = sd[prefix + ".running_var_h2"]

    eps = torch.tensor(eps)
    nf0 = torch.rsqrt(var0 + eps)
    nf1 = torch.rsqrt(var1 + eps)
    nf2 = torch.rsqrt(var2 + eps)

    s0 = ft0 * (1.0 - mu0) * nf0 - ft1 * mu1 * nf1 - ft2 * ((1.0 / math.sqrt(2.0)) + mu2) * nf2

    s1 = ft1 * nf1
    s2 = ft2 * (1.0 / math.sqrt(2.0)) * nf2

    a0 = gamma * s0 + beta
    a1 = gamma * s1
    a2 = gamma * s2

    return a0.numpy(), a1.numpy(), a2.numpy()


def pack():
    sd = torch.load(WEIGHT_PATH, map_location="cpu")

    N = 8192

    def dump(suffix, tp, obj):
        c = zstd.ZstdCompressor(level=6, threads=-1)
        with open(f"{BASE_DIR}/raw/{suffix}.pkl.zst", "wb") as f:
            with c.stream_writer(f) as out:
                pickle.dump((tp, obj), out, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{BASE_DIR}/ready/{suffix}.pkl.zst", "wb") as f:
            with c.stream_writer(f) as out:
                pickle.dump((tp, obj.pack(N)), out, protocol=pickle.HIGHEST_PROTOCOL)

    def conv1(kernel, bias, scale=1.0):
        lt = pack_conv2d_toeplitz(
            in_order=Order(batchsize=(64, 1, 1), C=1, H=32, W=32, H_real=28, W_real=28, tilesize=(4, 4)),
            out_order=Order(batchsize=(1, 64, 1), C=32, H=32, W=32, H_real=28, W_real=28, tilesize=(4, 4)),
            stride=1,
            kernel=kernel,
            bn_b=bias,
            out_scale=scale,
        )
        dump("conv1", Layer.LINEARTRANSFORM, lt)

    def act1(a0, a1, a2):
        order = Order(batchsize=(1, 64, 1), C=32, H=32, W=32, H_real=28, W_real=28, tilesize=(4, 4))
        hp = pack_herpn(order, a0, a1 / np.sqrt(a2), a2)
        dump("act1", Layer.HERPN, hp)

    def pool1(scale=1.0):
        lt = pack_avgpool2d(
            in_order=Order(batchsize=(1, 64, 1), C=32, H=32, W=32, H_real=28, W_real=28, tilesize=(4, 4)),
            out_order=Order(
                batchsize=(1, 64, 1), C=32, H=16, W=16, H_real=14, W_real=14, tilesize=(2, 2), gap=(1, 1)
            ),
            size=2,
            stride=2,
            padding=0,
            out_scale=scale,
        )
        dump("pool1", Layer.LINEARTRANSFORM, lt)

    def conv2(kernel, bias, scale=1.0):
        wino = pack_conv2d_winograd(
            in_order=Order(
                batchsize=(1, 64, 1), C=32, H=16, W=16, H_real=14, W_real=14, tilesize=(2, 2), gap=(1, 1)
            ),
            out_order=Order(
                batchsize=(1, 64, 1), C=64, H=16, W=16, H_real=14, W_real=14, tilesize=(2, 2), gap=(1, 1)
            ),
            kernel=kernel,
            bn_b=bias,
            out_scale=scale,
        )
        dump("conv2", Layer.WINOGRAD, wino)

    def act2(a0, a1, a2):
        order = Order(
            batchsize=(1, 64, 1), C=64, H=16, W=16, H_real=14, W_real=14, tilesize=(2, 2), gap=(1, 1)
        )
        hp = pack_herpn(order, a0, a1 / np.sqrt(a2), a2)
        dump("act2", Layer.HERPN, hp)

    def pool2(scale=1.0):
        lt = pack_avgpool2d(
            in_order=Order(
                batchsize=(1, 64, 1), C=64, H=16, W=16, H_real=14, W_real=14, tilesize=(2, 2), gap=(1, 1)
            ),
            out_order=Order(
                batchsize=(1, 64, 1), C=64, H=8, W=8, H_real=7, W_real=7, tilesize=(1, 1), gap=(1, 1)
            ),
            size=2,
            stride=2,
            padding=0,
            out_scale=scale,
        )
        dump("pool2", Layer.LINEARTRANSFORM, lt)

    def fc1(weight, bias, scale=1.0):
        scale = np.pad(scale, (0, 60))  # type: ignore[arg-type]
        lt = pack_fc(
            in_order=Order(
                batchsize=(1, 64, 1), C=64, H=8, W=8, H_real=7, W_real=7, tilesize=(1, 1), gap=(1, 1)
            ),
            out_order=Order(batchsize=(1, 64, 1), C=256, C_real=196, H=1, W=1, tilesize=(1, 1), gap=(64, 1)),
            weight=weight,
            bias=bias,
            out_scale=scale,  # type: ignore[arg-type]
        )
        dump("fc1", Layer.LINEARTRANSFORM, lt)

    def act3(a0, a1, a2):
        order = Order(batchsize=(1, 64, 1), C=256, C_real=196, H=1, W=1, tilesize=(1, 1), gap=(64, 1))
        hp = pack_herpn(order, a0, a1 / np.sqrt(a2), a2)
        dump("act3", Layer.HERPN, hp)

    def fc2(weight, bias, scale=1.0):
        lt = pack_fc(
            in_order=Order(batchsize=(1, 64, 1), C=256, C_real=196, H=1, W=1, gap=(64, 1)),
            out_order=Order(batchsize=(64, 1, 1), C=10, H=1, W=1, gap=(1, 1)),
            weight=weight,
            bias=bias,
            out_scale=scale,
        )
        dump("fc2", Layer.LINEARTRANSFORM, lt)

    kernel1, bias1 = sd["conv1.weight"].numpy(), sd["conv1.bias"].numpy()
    kernel2, bias2 = sd["conv2.weight"].numpy(), sd["conv2.bias"].numpy()
    act1_a0, act1_a1, act1_a2 = herpn_eval(sd, "act1")
    act2_a0, act2_a1, act2_a2 = herpn_eval(sd, "act2")
    act3_a0, act3_a1, act3_a2 = herpn_eval(sd, "act3")
    fc1_weight, fc1_bias = sd["fc1.weight"].numpy(), sd["fc1.bias"].numpy()
    fc2_weight, fc2_bias = sd["fc2.weight"].numpy(), sd["fc2.bias"].numpy()

    conv1(kernel1, bias1, np.sqrt(act1_a2))
    act1(act1_a0, act1_a1, act1_a2)
    pool1()
    conv2(kernel2, bias2, np.sqrt(act2_a2))
    act2(act2_a0, act2_a1, act2_a2)
    pool2()
    fc1(fc1_weight, fc1_bias, np.sqrt(act3_a2))
    act3(act3_a0, act3_a1, act3_a2)
    fc2(fc2_weight, fc2_bias)

    def find_rotations():
        ready_dir = Path(BASE_DIR) / "ready"
        files = sorted(ready_dir.glob("*.pkl.zst"))

        global_rots = set()
        for path in files:
            d = zstd.ZstdDecompressor()
            with path.open("rb") as f:
                with d.stream_reader(f) as src:
                    tp, obj = pickle.load(src)

            if tp == Layer.LINEARTRANSFORM:
                global_rots.update(obj.global_rots)
                if obj.output_rotations > 0:
                    i, offset = 0, N
                    while i < obj.output_rotations:
                        global_rots.add(offset >> 1)
                        i += 1
                        offset >>= 1

            elif tp == Layer.WINOGRAD:
                s = {int(i) for i in obj.i2c_offset.flatten() if i != 0}
                global_rots.update(s)
                global_rots.update(obj.global_rots)
                if obj.output_rotations > 0:
                    i, offset = 0, N
                    while i < obj.output_rotations:
                        global_rots.add(offset >> 1)
                        i += 1
                        offset >>= 1

        global_rots = [i % N for i in global_rots]

        converter = partial(
            preprocess,
            pad=((0, 0), (0, 0), (0, 4), (0, 4)),
            batchsize=(64, 1, 1),
            C=1,
            H=32,
            W=32,
            H_real=28,
            W_real=28,
            tilesize=(4, 4),
            gap=(1, 1),
        )

        path = Path(f"{BASE_DIR}/ready/meta.pkl")
        try:
            with path.open("rb") as f:
                meta = pickle.load(f)
        except FileNotFoundError:
            meta = {}
        meta["rots"] = sorted(global_rots)
        meta["converter"] = converter
        with path.open("wb") as f:
            pickle.dump(meta, f)

    find_rotations()


if __name__ == "__main__":
    pack()
