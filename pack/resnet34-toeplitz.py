#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 resnet34-toeplitz.py
# dir  := .
# kid  :=

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

from sagittarius.core import (
    Batchnorm2d,
    Layer,
    Order,
    pack_avgpool2d,
    pack_conv2d_toeplitz,
    pack_nonlinear,
    pack_fc,
    preprocess,
)

BASE_DIR = "../data/resnet34-toeplitz"

Path(f"{BASE_DIR}/raw").mkdir(parents=True, exist_ok=True)
Path(f"{BASE_DIR}/ready").mkdir(parents=True, exist_ok=True)

path = Path("/home/jyh/project/winograd/data/resnet34-toeplitz/ready/meta.pkl")
with path.open("rb") as f:
    statistics: dict = pickle.load(f)["statistics"]

model = torch.jit.load(
    "/home/jyh/project/winograd/clear/resnet34/models/resnet34_silu_avgpool_deploy.pt", map_location="cpu"
)
model.eval()
sd = model.state_dict()

for k in sd.keys():
    sd[k] = sd[k].double()

def pack():
    N = 32768

    def dump(suffix, tp, obj):
        c = zstd.ZstdCompressor(level=6, threads=-1)
        with open(f"{BASE_DIR}/raw/{suffix}.pkl.zst", "wb") as f:
            with c.stream_writer(f) as out:
                pickle.dump((tp, obj), out, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{BASE_DIR}/ready/{suffix}.pkl.zst", "wb") as f:
            with c.stream_writer(f) as out:
                pickle.dump((tp, obj.pack(N)), out, protocol=pickle.HIGHEST_PROTOCOL)

    def load_bn(prefix):
        prefix = f"backbone.{prefix}"

        def resolve_submodule(root, path: str):
            m = root
            for name in path.split("."):
                m = getattr(m, name)
            return m

        bn = resolve_submodule(model, prefix)
        BN = Batchnorm2d(
            mean=bn.running_mean.detach().double().numpy(),
            var=bn.running_var.detach().double().numpy(),
            gamma=bn.weight.detach().double().numpy(),
            beta=bn.bias.detach().double().numpy(),
            eps=bn.eps,
        )
        return BN.a, BN.b

    def load_kernel(name):
        kernel = sd[f"backbone.{name}.weight"].numpy()
        return kernel

    def silu(x):
        return x / (1 + np.exp(-x))

    def conv1():
        print("== Packing conv1 ==")
        kernel = load_kernel("conv1")
        bn_a, bn_b = load_bn("bn1")
        lt = pack_conv2d_toeplitz(
            in_order=Order(C=3, H=224, W=224, gap=(1, 1)),
            out_order=Order(C=64, H=112, W=112, gap=(2, 2)),
            stride=2,
            kernel=kernel,
            bn_a=bn_a,  # type: ignore[arg-type]
            bn_b=bn_b,  # type: ignore[arg-type]
        )
        dump("conv1", Layer.LINEARTRANSFORM, lt)

    def pack_silu(name, degree, sample):
        xmin, xmax = statistics[sample]["min"], statistics[sample]["max"]
        obj = pack_nonlinear(silu, degree, a=xmin, b=xmax)
        dump(name, Layer.NONLINEAR, obj)

    def pool1():
        print("== Packing pool1 ==")
        lt = pack_avgpool2d(
            in_order=Order(C=64, H=112, W=112, gap=(2, 2)),
            out_order=Order(C=64, H=56, W=56, gap=(4, 4)),
            padding=1,
            size=3,
            stride=2,
        )
        dump("pool1", Layer.LINEARTRANSFORM, lt)

    def layer1():
        layer = 1
        for block in range(3):
            print(f"== Packing layer {layer} block {block} ==")
            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn1")
            lt = pack_conv2d_toeplitz(
                in_order=Order(C=64, H=56, W=56, gap=(4, 4)),
                out_order=Order(C=64, H=56, W=56, gap=(4, 4)),
                stride=1,
                kernel=load_kernel(f"layer{layer}.{block}.conv1"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv1", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act1", 63, f"layer{layer}.{block}.1")

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn2")
            lt = pack_conv2d_toeplitz(
                in_order=Order(C=64, H=56, W=56, gap=(4, 4)),
                out_order=Order(C=64, H=56, W=56, gap=(4, 4)),
                stride=1,
                kernel=load_kernel(f"layer{layer}.{block}.conv2"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv2", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act2", 63, f"layer{layer}.{block}.2")

    def layer2():
        layer = 2
        order_pre = Order(C=64, H=56, W=56, gap=(4, 4))
        order_cur = Order(C=128, H=28, W=28, gap=(8, 8))
        for block in range(4):
            print(f"== Packing layer {layer} block {block} ==")
            if block == 0:
                bn_a, bn_b = load_bn(f"layer{layer}.{block}.downsample.1")
                lt = pack_conv2d_toeplitz(
                    in_order=order_pre,
                    out_order=order_cur,
                    stride=2,
                    kernel=load_kernel(f"layer{layer}.{block}.downsample.0"),
                    bn_a=bn_a,  # type: ignore[arg-type]
                    bn_b=bn_b,  # type: ignore[arg-type]
                )
                dump(f"layer{layer}.{block}.downsample", Layer.LINEARTRANSFORM, lt)

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn1")
            lt = pack_conv2d_toeplitz(
                in_order=order_cur if block > 0 else order_pre,
                out_order=order_cur,
                stride=1 if block > 0 else 2,
                kernel=load_kernel(f"layer{layer}.{block}.conv1"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv1", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act1", 63, f"layer{layer}.{block}.1")

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn2")
            lt = pack_conv2d_toeplitz(
                in_order=order_cur,
                out_order=order_cur,
                stride=1,
                kernel=load_kernel(f"layer{layer}.{block}.conv2"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv2", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act2", 63, f"layer{layer}.{block}.2")

    def layer3():
        layer = 3
        order_pre = Order(C=128, H=28, W=28, gap=(8, 8))
        order_cur = Order(C=256, H=14, W=14, gap=(16, 16))
        for block in range(6):
            print(f"== Packing layer {layer} block {block} ==")
            if block == 0:
                bn_a, bn_b = load_bn(f"layer{layer}.{block}.downsample.1")
                lt = pack_conv2d_toeplitz(
                    in_order=order_pre,
                    out_order=order_cur,
                    stride=2,
                    kernel=load_kernel(f"layer{layer}.{block}.downsample.0"),
                    bn_a=bn_a,  # type: ignore[arg-type]
                    bn_b=bn_b,  # type: ignore[arg-type]
                )
                dump(f"layer{layer}.{block}.downsample", Layer.LINEARTRANSFORM, lt)

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn1")
            lt = pack_conv2d_toeplitz(
                in_order=order_cur if block > 0 else order_pre,
                out_order=order_cur,
                stride=1 if block > 0 else 2,
                kernel=load_kernel(f"layer{layer}.{block}.conv1"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv1", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act1", 63, f"layer{layer}.{block}.1")

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn2")
            lt = pack_conv2d_toeplitz(
                in_order=order_cur,
                out_order=order_cur,
                stride=1,
                kernel=load_kernel(f"layer{layer}.{block}.conv2"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv2", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act2", 63, f"layer{layer}.{block}.2")

    def layer4():
        layer = 4
        order_pre = Order(C=256, H=14, W=14, gap=(16, 16))
        order_cur = Order(C=1024, C_real=512, H=7, W=7, gap=(32, 32))
        for block in range(3):
            print(f"== Packing layer {layer} block {block} ==")
            if block == 0:
                bn_a, bn_b = load_bn(f"layer{layer}.{block}.downsample.1")
                bn_a, bn_b = np.pad(bn_a, (0, 512)), np.pad(bn_b, (0, 512))
                lt = pack_conv2d_toeplitz(
                    in_order=order_pre,
                    out_order=order_cur,
                    stride=2,
                    kernel=load_kernel(f"layer{layer}.{block}.downsample.0"),
                    bn_a=bn_a,  # type: ignore[arg-type]
                    bn_b=bn_b,  # type: ignore[arg-type]
                )
                dump(f"layer{layer}.{block}.downsample", Layer.LINEARTRANSFORM, lt)

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn1")
            bn_a, bn_b = np.pad(bn_a, (0, 512)), np.pad(bn_b, (0, 512))
            lt = pack_conv2d_toeplitz(
                in_order=order_cur if block > 0 else order_pre,
                out_order=order_cur,
                stride=1 if block > 0 else 2,
                kernel=load_kernel(f"layer{layer}.{block}.conv1"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv1", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act1", 63, f"layer{layer}.{block}.1")

            bn_a, bn_b = load_bn(f"layer{layer}.{block}.bn2")
            bn_a, bn_b = np.pad(bn_a, (0, 512)), np.pad(bn_b, (0, 512))
            lt = pack_conv2d_toeplitz(
                in_order=order_cur,
                out_order=order_cur,
                stride=1,
                kernel=load_kernel(f"layer{layer}.{block}.conv2"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
            )
            dump(f"layer{layer}.{block}.conv2", Layer.LINEARTRANSFORM, lt)

            pack_silu(f"layer{layer}.{block}.act2", 63 if block < 5 else 31, f"layer{layer}.{block}.2")

    def pool2():
        print("== Packing pool2 ==")
        lt = pack_avgpool2d(
            in_order=Order(C=1024, C_real=512, H=7, W=7, gap=(32, 32)),
            out_order=Order(C=1024, C_real=512, H=1, W=1, gap=(1, 1)),
            size=7,
            stride=1,
            padding=0
        )
        dump("pool2", Layer.LINEARTRANSFORM, lt)

    def fc():
        print("== Packing fc ==")
        lt = pack_fc(
            in_order=Order(C=1024, C_real=512, H=1, W=1, gap=(1, 1)),
            out_order=Order(C=1000, H=1, W=1, gap=(1, 1)),
            weight=sd["backbone.fc.weight"].numpy(),
            bias=sd["backbone.fc.bias"].numpy(),
        )
        dump("fc", Layer.LINEARTRANSFORM, lt)

    # conv1()
    pack_silu("act1", 31, "conv1")
    pool1()
    layer1()
    layer2()
    layer3()
    layer4()
    pool2()
    fc()

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
            pad=((0, 0), (0, 0), (0, 0), (0, 0)),
            batchsize=(1, 1, 1),
            C=3,
            H=224,
            W=224,
            H_real=224,
            W_real=224,
            tilesize=(1, 1),
            gap=(1, 1),
        )

        path = Path("/home/jyh/project/winograd/data/resnet34-toeplitz/ready/meta.pkl")
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


pack()
