#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 resnet34-toeplitz.py
# dir  := .
# kid  :=

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import zstandard as zstd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sagittarius.core import (
    Batchnorm2d,
    Cipher,
    Layer,
    Order,
    num_diags,
    pack_conv2d_toeplitz,
    pack_nonlinear,
    pack_avgpool2d,
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

def pack():
    base_dir = BASE_DIR
    N = 32768
    CLAMP = 1.0

    def dump(suffix, tp, obj):
        c = zstd.ZstdCompressor(level=6, threads=-1)
        with open(f"{base_dir}/ready/{suffix}.pkl.zst", "wb") as f:
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
            mean=bn.running_mean.detach().numpy(),
            var=bn.running_var.detach().numpy(),
            gamma=bn.weight.detach().numpy(),
            beta=bn.bias.detach().numpy(),
            eps=bn.eps,
        )
        return BN.a, BN.b

    def load_scale(name):
        if name is None:
            return 1.0, 0.0
        xmin, xmax = statistics[name]["min"], statistics[name]["max"]
        scale = 2 * CLAMP / (xmax - xmin)
        offset = -CLAMP * (xmax + xmin) / (xmax - xmin)
        return scale, offset

    def load_kernel(name):
        kernel = sd[f"backbone.{name}.weight"].numpy()
        return kernel

    def load_conv(k, bn, inf=None, outf=None):
        kernel = load_kernel(k)
        bn_a, bn_b = load_bn(bn)
        in_scale, in_offset = (1.0, 0.0) if inf is None else load_scale(inf)
        out_scale, out_offset = (1.0, 0.0) if outf is None else load_scale(outf)
        return kernel, bn_a, bn_b, in_scale, in_offset, out_scale, out_offset

    def silu(x):
        return x / (1 + np.exp(-x))

    def conv1():
        kernel, bn_a, bn_b, in_scale, in_offset, out_scale, out_offset = load_conv(
            "conv1", "bn1", None, "conv1"
        )
        lt = pack_conv2d_toeplitz(
            in_order=Order(C=3, H=224, W=224, gap=(1, 1)),
            out_order=Order(C=64, H=112, W=112, gap=(2, 2)),
            stride=2,
            kernel=kernel,
            bn_a=bn_a,  # type: ignore[arg-type]
            bn_b=bn_b,  # type: ignore[arg-type]
            # out_scale=out_scale,
            # out_offset=out_offset,  # type: ignore[arg-type]
            # in_scale=in_scale,
            # in_offset=in_offset,  # type: ignore[arg-type]
        )
        dump("conv1", Layer.LINEARTRANSFORM, lt)

    # def pack_silu(name, degree, inf=None, outf=None):
    #     in_scale, in_offset = load_scale(inf)
    #     out_scale, out_offset = load_scale(outf)
    #     # obj = pack_nonlinear(silu, degree, in_scale, in_offset, out_offset, out_scale)
    #     obj = pack_nonlinear(silu, degree)
    #     dump(name, Layer.NONLINEAR, obj)

    def pack_silu(name, degree, sample):
        xmin, xmax = statistics[sample]["min"], statistics[sample]["max"]
        obj = pack_nonlinear(silu, degree, a=xmin, b=xmax)
        dump(name, Layer.NONLINEAR, obj)


    def pool1():
        out_scale, out_offset = load_scale("layer1.0.2")
        lt = pack_avgpool2d(
            in_order=Order(C=64, H=112, W=112, gap=(2, 2)),
            out_order=Order(C=64, H=56, W=56, gap=(4, 4)),
            size=3,
            stride=2,
            # out_scale=out_scale * 0.5,
            # out_offset=out_offset * 0.5
        )
        dump("pool1", Layer.LINEARTRANSFORM, lt)

    def layer1():
        in_order = Order(C=64, H=56, W=56, gap=(4, 4))
        out_order = Order(C=64, H=56, W=56, gap=(4, 4))
        for block in range(3):
            scale1, offset1 = load_scale(f"layer1.{block}.1")
            scale2, offset2 = load_scale(f"layer1.{block}.2")
            bn_a, bn_b = load_bn(f"layer1.{block}.bn1")
            lt = pack_conv2d_toeplitz(
                in_order=in_order,
                out_order=out_order,
                stride=1,
                kernel=load_kernel(f"layer1.{block}.conv1"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
                # in_scale=scale2 * 0.5,
                # in_offset=offset2 * 0.5,
                # out_scale=scale1,
                # out_offset=offset1,
            )

            act = pack_nonlinear(silu, 63, in_scale=scale1, in_offset=offset1)
            dump(f"layer1.{block}.act1", Layer.NONLINEAR, act)

            dump(f"layer1.{block}.conv1", Layer.LINEARTRANSFORM, lt)
            bn_a, bn_b = load_bn(f"layer1.{block}.bn2")
            lt = pack_conv2d_toeplitz(
                in_order=in_order,
                out_order=out_order,
                stride=1,
                kernel=load_kernel(f"layer1.{block}.conv1"),
                bn_a=bn_a,  # type: ignore[arg-type]
                bn_b=bn_b,  # type: ignore[arg-type]
                # in_scale=scale1,
                # in_offset=offset1,
                # out_scale=scale2 * 0.5,
                # out_offset=offset2 * 0.5,
            )
            dump(f"layer1.{block}.conv2", Layer.LINEARTRANSFORM, lt)

            scale3, offset3 = load_scale(f"layer1.{block+1}.2") if block + 1 < 3 else load_scale(None)
            act = pack_nonlinear(silu, 63, in_scale=scale2 * 0.5, in_offset=offset2 * 0.5, out_scale=scale3 * 0.5, out_offset=offset3 * 0.5)
            dump(f"layer1.{block}.act2", Layer.NONLINEAR, act)
            break


    conv1()
    pack_silu("act1", 31, "conv1")
    pool1()
    layer1()

    def find_rotations():
        ready_dir = Path(base_dir) / "ready"
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
