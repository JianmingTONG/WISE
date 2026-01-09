#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 mnist.py
# run  := cd /home/jyh/project/winograd/backend/openfhe/build/ && cmake -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR=$HOME/opt/openfhe-v1.4.0/lib/OpenFHE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && cmake --build . -j 32 && cd /home/jyh/project/winograd/runner && python3 mnist.py
# dir  := .
# kid  :=

import sys
import importlib.util
spec = importlib.util.spec_from_file_location('fhecore', '/home/ubuntu/WISE/backend/openfhe/bindings/python/build/fhecore.cpython-313-x86_64-linux-gnu.so')
fhecore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fhecore)

import numpy as np
import tomllib
from dataclasses import dataclass
import pickle
from typing import List
from sagittarius.core import pad_to_kN, Layer
import zstandard as zstd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["winograd", "toeplitz", "toeplitz-pad"],
        required=True,
        help="Convolution packing",
    )
    return parser.parse_args()


args = parse_args()
if args.mode == "winograd":
    BASE_DIR = "../data/mnist-winograd"
elif args.mode == "toeplitz":
    BASE_DIR = "../data/mnist-toeplitz"
elif args.mode == "toeplitz-pad":
    BASE_DIR = "../data/mnist-toeplitz-pad"


BATCH_SIZE = 64
CONFIG_PATH = "../config/mnist.toml"

def fetch():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(root="../clear/data", train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    for batch in test_loader:
        yield batch

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
    composite_degree: int
    register_word_size: int

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
        composite_degree=rf.get("composite_degree", 1),
        register_word_size=rf.get("register_word_size", 0),
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
    params.composite_degree = cfg.fhe.composite_degree
    params.register_word_size = cfg.fhe.register_word_size

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

def eval(input_raw):

    d = zstd.ZstdDecompressor()

    def eval_conv2d(fn, level):
        with open(f"{BASE_DIR}/ready/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.LINEARTRANSFORM:
            return ctx.eval_linear_transform(x, obj, level)
        elif tp == Layer.WINOGRAD:
            return ctx.eval_winograd(x, obj, level)
        else:
            raise ValueError("Unknown layer type")

    def eval_activation(fn, level):
        with open(f"{BASE_DIR}/ready/{fn}.pkl.zst", "rb") as f:
            with d.stream_reader(f) as src:
                tp, obj = pickle.load(src)
        if tp == Layer.HERPN:
            return ctx.eval_herpn(x, obj, level)
        else:
            raise ValueError("Unknown activation type")

    x = ctx.encrypt_batch(input_raw.reshape(-1, N), 0)

    # Level progression: 0→1→2→3→4→5→6→7→8
    # With composite scaling (composite_degree > 1), each "level" consists of
    # composite_degree primes. The level count stays the same; only the internal
    # representation changes (each level = product of multiple small primes).
    x = eval_conv2d("conv1", 0)
    x = eval_activation("act1", 1)
    x = eval_conv2d("pool1", 2)
    x = eval_conv2d("conv2", 3)
    x = eval_activation("act2", 4)
    x = eval_conv2d("pool2", 5)
    x = eval_conv2d("fc1", 6)
    x = eval_activation("act3", 7)
    x = eval_conv2d("fc2", 8)

    x = ctx.decrypt_batch(x)
    return x

it = fetch()
data, label = next(it)
x = pad_to_kN(converter(data.numpy()), N)

y_he = eval(x).flatten()[:64*10].reshape(64, 10)
label_he = np.argmax(y_he, axis=1)
acc = (label_he == label.numpy()).mean()
print(f"HE Accuracy: {acc*100:.2f}%")
