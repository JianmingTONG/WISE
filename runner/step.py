#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 step.py --mode winograd
# run  := python3 resnet34.py --mode winograd
# run  := cd /home/jyh/project/winograd/backend/openfhe/build/ && cmake -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR=$HOME/opt/openfhe-v1.4.0/lib/OpenFHE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && cmake --build . -j 32 && cd /home/jyh/project/winograd/runner && python3 mnist.py
# dir  := .
# kid  :=

import fhecore

import numpy as np
import tomllib
from dataclasses import dataclass
import pickle
from typing import List
from sagittarius.core import pad_to_kN, Layer
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
import torch
import zstandard as zstd
import argparse
import math

CONFIG_PATH = "../config/resnet34.toml"

def silu(x):
    return x / (1 + np.exp(-x))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["winograd", "toeplitz"],
        required=True,
        help="Convolution packing",
    )
    return parser.parse_args()


args = parse_args()
if args.mode == "toeplitz":
    BASE_DIR = "../data/resnet34-toeplitz"
    TEMP_DIR = "temp-toeplitz"
elif args.mode == "winograd":
    BASE_DIR = "../data/resnet34-winograd"
    TEMP_DIR = "temp-winograd"


def fetch(batchsize=64, dset="train", seed=None):
    root = "../clear/resnet34/dataset"
    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    ds = ImageFolder(f"{root}/{dset}", transform=preprocess)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    images, labels = next(iter(loader))
    return images, labels

@dataclass
class FheCfg:
    log_N: int
    scale_mod_size: int
    first_mod_size: int
    log_batch_size: int
    mult_depth: int
    secret_key_dist: str
    scaling_technique: str
    security_level: str
    level_budget: List[int]

@dataclass
class Cfg:
    fhe: FheCfg

def load_cfg(path=CONFIG_PATH) -> Cfg:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    rf = raw["fhe"]
    fhe = FheCfg(
        log_N=rf["log_N"],
        scale_mod_size=rf["scale_mod_size"],
        first_mod_size=rf["first_mod_size"],
        log_batch_size=rf["log_batch_size"],
        mult_depth=rf["mult_depth"],
        secret_key_dist=rf["secret_key_dist"],
        scaling_technique=rf["scaling_technique"],
        security_level=rf["security_level"],
        level_budget=rf.get("level_budget"),
    )
    return Cfg(fhe=fhe)


cfg = load_cfg()

def init():
    with open(f"{BASE_DIR}/ready/meta.pkl", "rb") as f:
        meta = pickle.load(f)
        global_rots = meta["rots"]
        converter = meta["converter"]

    params = fhecore.FHEParams()
    params.log_N = cfg.fhe.log_N
    params.scale_mod_size = cfg.fhe.scale_mod_size
    params.first_mod_size = cfg.fhe.first_mod_size
    params.mult_depth = cfg.fhe.mult_depth
    params.log_batch_size = cfg.fhe.log_batch_size
    params.secret_key_dist = cfg.fhe.secret_key_dist
    params.scaling_technique = cfg.fhe.scaling_technique
    params.security_level = cfg.fhe.security_level
    params.level_budget = cfg.fhe.level_budget
    params.global_rots = global_rots

    ctx = fhecore.FHEContext(params)
    return ctx, converter

N = 1 << cfg.fhe.log_batch_size
ctx, converter = init()

def cipherprint(cs, size=10):
    p = ctx.decrypt_batch(cs)
    print(p.flatten()[:size])

def stats(x):
    level, _, scaling_factor, num_primes, levels_remaining = ctx.stats(x)
    print(f"level = {level}, scaling_factor = {scaling_factor}, num_primes = {num_primes}, levels_remaining = {levels_remaining}")

def eval_clear(input_raw):
    d = zstd.ZstdDecompressor()

    def eval_conv2d(x, fn):
        with open(f"{BASE_DIR}/raw/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.LINEARTRANSFORM:
            a, b = obj.mat.shape
            a, b = math.ceil(a / N) * N, math.ceil(b / N) * N
            obj.mat.resize(a, b)
            obj.bias = pad_to_kN(obj.bias, N)
            return obj.mat @ x + obj.bias
        elif tp == Layer.WINOGRAD:
            raise ValueError("Unknown layer type")
        else:
            raise ValueError("Unknown layer type")

    def eval_linear_transform(x, fn):
        with open(f"{BASE_DIR}/raw/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.LINEARTRANSFORM:
            a, b = obj.mat.shape
            a, b = math.ceil(a / N) * N, math.ceil(b / N) * N
            obj.mat.resize(a, b)
            obj.bias = pad_to_kN(obj.bias, N)
            return obj.mat @ x + obj.bias
        else:
            raise ValueError("Unknown layer type")

    def eval_activation(x):
        return silu(x)

    x = input_raw
    return x
    x = eval_conv2d(x, "conv1")
    x = eval_activation(x)
    x = eval_linear_transform(x, "pool1")

    # NOTE: Layer 1
    for block in range(3):
        layer = 1
        skip = x.copy()
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1")
        x = eval_activation(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2")
        x = x + skip
        x = eval_activation(x)

    # NOTE: Layer 2
    for block in range(4):
        layer = 2
        if block == 0:
            skip = eval_conv2d(x, f"layer{layer}.{block}.downsample")
        else:
            skip = x.copy()
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1")
        x = eval_activation(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2")
        x = x + skip
        x = eval_activation(x)

    # NOTE: Layer 3
    for block in range(6):
        layer = 3
        if block == 0:
            skip = eval_conv2d(x, f"layer{layer}.{block}.downsample")
        else:
            skip = x.copy()
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1")
        x = eval_activation(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2")
        x = x + skip
        x = eval_activation(x)

    # NOTE: Layer 4
    for block in range(3):
        layer = 4
        if block == 0:
            skip = eval_conv2d(x, f"layer{layer}.{block}.downsample")
        else:
            skip = x.copy()
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1")
        x = eval_activation(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2")
        x = x + skip
        x = eval_activation(x)

    # NOTE: final
    x = eval_linear_transform(x, "pool2")
    x = eval_linear_transform(x, "fc")

    return x

def eval(input_raw):

    depth = ctx.depth()

    d = zstd.ZstdDecompressor()

    def eval_conv2d(x, fn, level):
        with open(f"{BASE_DIR}/ready/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.LINEARTRANSFORM:
            return ctx.eval_linear_transform(x, obj, level)
        elif tp == Layer.WINOGRAD:
            return ctx.eval_winograd(x, obj, level)
        else:
            raise ValueError("Unknown layer type")

    def eval_linear_transform(x, fn, level):
        with open(f"{BASE_DIR}/ready/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.LINEARTRANSFORM:
            return ctx.eval_linear_transform(x, obj, level)
        else:
            raise ValueError("Unknown layer type")

    def eval_activation(x, fn, level):
        with open(f"{BASE_DIR}/ready/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.HERPN:
            return ctx.eval_herpn(x, obj, level)
        elif tp == Layer.NONLINEAR:
            return ctx.eval_nonlinear(x, obj, level)
        else:
            raise ValueError("Unknown activation type")

    x = ctx.encrypt_batch(input_raw.reshape(-1, N), depth-2)

    x = eval_conv2d(x, "conv1", depth-2)
    x = ctx.eval_bootstrap_batch(x)
    x = eval_activation(x, "act1", depth-10)
    x = eval_linear_transform(x, "pool1", depth-3)

    x = ctx.decrypt_batch(x); return x;

    # NOTE: Layer 1
    for block in range(3):
        layer = 1
        skip = ctx.deepcopy_ciphertexts(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = eval_activation(x, f"layer{layer}.{block}.act1", depth-10)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2", depth-2)
        x = ctx.eval_add_batch(x, skip)
        x = ctx.eval_bootstrap_batch(x)
        x = eval_activation(x, f"layer{layer}.{block}.act2", depth-10)

    # NOTE: Layer 2
    for block in range(4):
        layer = 2
        if block == 0:
            skip = eval_conv2d(x, f"layer{layer}.{block}.downsample", depth-2)
        else:
            skip = ctx.deepcopy_ciphertexts(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = eval_activation(x, f"layer{layer}.{block}.act1", depth-10)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = ctx.eval_add_batch(x, skip)
        x = eval_activation(x, f"layer{layer}.{block}.act2", depth-10)

    # NOTE: Layer 3
    for block in range(6):
        layer = 3
        if block == 0:
            skip = eval_conv2d(x, f"layer{layer}.{block}.downsample", depth-2)
        else:
            skip = ctx.deepcopy_ciphertexts(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = eval_activation(x, f"layer{layer}.{block}.act1", depth-10)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = ctx.eval_add_batch(x, skip)
        x = eval_activation(x, f"layer{layer}.{block}.act2", depth-10)

    # NOTE: Layer 4
    for block in range(3):
        layer = 4
        if block == 0:
            skip = eval_conv2d(x, f"layer{layer}.{block}.downsample", depth-2)
        else:
            skip = ctx.deepcopy_ciphertexts(x)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv1", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = eval_activation(x, f"layer{layer}.{block}.act1", depth-10)
        x = eval_conv2d(x, f"layer{layer}.{block}.conv2", depth-2)
        x = ctx.eval_bootstrap_batch(x)
        x = ctx.eval_add_batch(x, skip)
        x = eval_activation(x, f"layer{layer}.{block}.act2", depth-10)

    # NOTE: final
    x = eval_linear_transform(x, "pool2", depth-3)
    x = eval_linear_transform(x, "fc", depth-2)

    x = ctx.decrypt_batch(x)
    return x

def statistic(x):
    print(f"shape: {x.shape}, min: {np.min(x)}, max: {np.max(x)}, mean: {np.mean(x)}, std: {np.std(x)}")

data, label = fetch(batchsize=1, seed=42)
x = pad_to_kN(converter(data.numpy()), N)
x = eval_clear(x)
# y_he = eval(x)
# y_he = y_he.flatten()
# print(y_he)
# statistic(y_he)
