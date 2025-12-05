#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 resnet34_winograd_pack.py
# dir  := .
# kid  :=

import torch
from sagittarius.utils import (
    Batchnorm2d,
    Layer,
    Order,
    pack_avgpool2d,
    pack_conv2d_toeplitz,
    pack_conv2d_winograd,
)
import pickle

def resnet34():
    import torchvision
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model.eval()

    sd = model.state_dict()

    base_dir = "/home/jyh/project/winograd/data/resnet34-winograd"

    def load_kernel(prefix):
        return sd[f"{prefix}.weight"]

    def load_bn(prefix):
        bn = model.get_submodule(prefix)
        assert isinstance(bn, torch.nn.BatchNorm2d)
        assert isinstance(bn.running_mean, torch.Tensor)
        assert isinstance(bn.running_var, torch.Tensor)
        return Batchnorm2d(
            mean=bn.running_mean.detach().numpy(),
            var=bn.running_var.detach().numpy(),
            gamma=bn.weight.detach().numpy(),
            beta=bn.bias.detach().numpy(),
            eps=bn.eps
        )

    N = 32768

    def conv1():
        lt = pack_conv2d_toeplitz(
            in_order=Order(C=3, H=256, W=256, H_real=224, W_real=224, Ht=2, Wt=2),
            out_order=Order(C=64, H=128, W=128, H_real=112, W_real=112, Ht=2, Wt=2, gap=2),
            stride=2,
            kernel=load_kernel("conv1"),
            batchnorm=load_bn("bn1")
        )
        with open(f"{base_dir}/raw/conv1.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt), f)
        with open(f"{base_dir}/ready/conv1.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt.pack(N)), f)

    def pool1():
        lt = pack_avgpool2d(
            in_order=Order(C=64, H=128, W=128, H_real=112, W_real=112, Ht=2, Wt=2, gap=2),
            out_order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            size=3,
            stride=2
        )
        with open(f"{base_dir}/raw/pool1.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt), f)
        with open(f"{base_dir}/ready/pool1.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt.pack(N)), f)

    def layer1_0_conv1():
        prefix, cid = "layer1.0", 1
        wino = pack_conv2d_winograd(
            order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        print("Start writing")
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer1_0_conv2():
        prefix, cid = "layer1.0", 2
        wino = pack_conv2d_winograd(
            order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer1_1_conv1():
        prefix, cid = "layer1.1", 1
        wino = pack_conv2d_winograd(
            order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer1_1_conv2():
        prefix, cid = "layer1.1", 2
        wino = pack_conv2d_winograd(
            order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer1_2_conv1():
        prefix, cid = "layer1.2", 1
        wino = pack_conv2d_winograd(
            order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer1_2_conv2():
        prefix, cid = "layer1.2", 2
        wino = pack_conv2d_winograd(
            order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)


    def layer2_0_downsample():
        lt = pack_conv2d_toeplitz(
            in_order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            out_order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            stride=2,
            kernel=load_kernel("layer2.0.downsample.0"),
            batchnorm=load_bn("layer2.0.downsample.1")
        )
        with open(f"{base_dir}/raw/layer2_0_downsample.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt), f)
        with open(f"{base_dir}/ready/layer2_0_downsample.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt.pack(N)), f)

    def layer2_0_conv1():
        prefix, cid = "layer2.0", 1
        lt = pack_conv2d_toeplitz(
            in_order=Order(C=64, H=64, W=64, H_real=56, W_real=56, Ht=2, Wt=2, gap=4),
            out_order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            stride=2,
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.LINEARTRANSFORM, lt.pack(N)), f)

    def layer2_0_conv2():
        prefix, cid = "layer2.0", 2
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer2_1_conv1():
        prefix, cid = "layer2.1", 1
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer2_1_conv2():
        prefix, cid = "layer2.1", 2
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer2_2_conv1():
        prefix, cid = "layer2.2", 1
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer2_2_conv2():
        prefix, cid = "layer2.2", 2
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer2_3_conv1():
        prefix, cid = "layer2.3", 1
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    def layer2_3_conv2():
        prefix, cid = "layer2.3", 2
        wino = pack_conv2d_winograd(
            order=Order(C=128, H=32, W=32, H_real=28, W_real=28, Ht=2, Wt=2, gap=8),
            kernel=load_kernel(f"{prefix}.conv{cid}"),
            batchnorm=load_bn(f"{prefix}.bn{cid}")
        )
        with open(f"{base_dir}/raw/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino), f)
        with open(f"{base_dir}/ready/{prefix.replace('.', '_')}_conv{cid}.pkl", "wb") as f:
            pickle.dump((Layer.WINOGRAD, wino.pack(N)), f)

    conv1()
    pool1()

    # layer1_0_conv1()
    # layer1_0_conv2()

    # layer1_1_conv1()
    # layer1_1_conv2()

    # layer1_2_conv1()
    # layer1_2_conv2()

    # layer2_0_downsample()
    # layer2_0_conv1()
    # layer2_0_conv2()

    # layer2_1_conv1()
    # layer2_1_conv2()

    # layer2_2_conv1()
    # layer2_2_conv2()

    # layer2_3_conv1()
    # layer2_3_conv2()

if __name__ == "__main__":
    resnet34()
