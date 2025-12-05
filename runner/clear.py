#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 clear.py
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
import math

BASE_DIR = "/home/jyh/project/winograd/data/resnet34-toeplitz"
CONFIG_PATH = "/home/jyh/project/winograd/config/resnet34.toml"

def fetch(batchsize=64, dset="train", seed=None):
    root="/home/jyh/project/winograd/clear/resnet34/dataset"
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
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    images, labels = next(iter(loader))
    return images.double(), labels

def init():
    with open(f"{BASE_DIR}/ready/meta.pkl", "rb") as f:
        meta = pickle.load(f)
        converter = meta["converter"]
    return converter

N = 32768
converter = init()

def silu(x):
    return x / (1 + np.exp(-x))

def eval(input_raw):
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

def statistic(x):
    print(f"shape: {x.shape}\nmin: {np.min(x)}, max: {np.max(x)}, mean: {np.mean(x)}, std: {np.std(x)}")

data, label = fetch(batchsize=1, seed=42)
x = pad_to_kN(converter(data.numpy()), N)

y_he = eval(x)
statistic(y_he)
# y_he = y_he.flatten()
# print(y_he)
# statistic(y_he)
