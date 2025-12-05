#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 silu.py
# run  := cd /home/jyh/project/winograd && CMAKE_ARGS="-DOpenFHE_DIR=$HOME/opt/openfhe-v1.4.0/lib/OpenFHE" pip install -e backend/openfhe/bindings/python && cd /home/jyh/project/winograd/runner && python3 silu.py
# dir  := .
# kid  :=

import fhecore

import numpy as np
import tomllib
from dataclasses import dataclass
import pickle
from typing import List
from sagittarius.core import pad_to_kN, Layer

BASE_DIR = "../data/silu-test"
CONFIG_PATH = "../config/silu.toml"

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
        input_raw = meta["input"]

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
    return ctx, converter, input_raw

N = 1 << cfg.fhe.log_batch_size
ctx, converter, input_raw = init()

def cipherprint(cs, size=10):
    p = ctx.decrypt_batch(cs)
    print(p.flatten()[:size])

def stats(x):
    level, _, scaling_factor, num_primes, levels_remaining = ctx.stats(x)
    print(f"level = {level}, scaling_factor = {scaling_factor}, num_primes = {num_primes}, levels_remaining = {levels_remaining}")

def eval(input_raw):

    def eval_conv2d(fn, level):
        tp, obj = pickle.load(open(f"{BASE_DIR}/ready/{fn}.pkl", "rb"))
        if tp == Layer.LINEARTRANSFORM:
            return ctx.eval_linear_transform(x, obj, level)
        elif tp == Layer.WINOGRAD:
            return ctx.eval_winograd(x, obj, level)
        else:
            raise ValueError("Unknown layer type")

    def eval_activation(fn, level):
        tp, obj = pickle.load(open(f"{BASE_DIR}/ready/{fn}.pkl", "rb"))
        if tp == Layer.HERPN:
            return ctx.eval_herpn(x, obj, level)
        elif tp == Layer.NONLINEAR:
            return ctx.eval_nonlinear(x, obj, level)
        else:
            raise ValueError("Unknown activation type")

    x = ctx.encrypt_batch(input_raw.reshape(-1, N), 0)

    # x = eval_conv2d("conv1", 0); stats(x[0])
    x = eval_conv2d("pool1", 0); stats(x[0])
    x = ctx.eval_bootstrap_batch(x)
    # x = eval_conv2d("fc1", 0); stats(x[0])
    # x = eval_activation("silu1", 0)

    # x = eval_activation("act1", 1)
    # stats(x[0])

    x = ctx.decrypt_batch(x)
    print(f"shape: {x.shape}\nmin: {np.min(x)}, max: {np.max(x)}, mean: {np.mean(x)}, std: {np.std(x)}")
    return x

eval(input_raw)
