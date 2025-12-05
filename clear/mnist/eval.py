#!/home/jyh/project/winograd/winograd-venv/bin/python3
# run  := python3 eval.py
# dir  := .
# kid  :=

import math

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


batch_size = 64

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

sd = torch.load("/home/jyh/project/mnist/imp/mnist_cnn_herpn_avgpool_wo_padding_196.pth", map_location="cpu")

def herpn_eval(prefix):
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

    print(ft2)
    print(ft1)
    print(ft0)
    exit(0)

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

    return a0, a1, a2


# def stat(x, padded):
#     x = np.pad(
#         x.numpy(),
#         ((0, 0), (0, 0), (0, padded - x.shape[2]), (0, padded - x.shape[3])),
#         "constant",
#         constant_values=0,
#     )
#     print(f"mean={x.mean()}")
#     print(f"min={x.min()}")
#     print(f"max={x.max()}")
#     print(f"std={x.std()}")


# def stat_raw(x):
#     print(f"mean={x.mean()}")
#     print(f"min={x.min()}")
#     print(f"max={x.max()}")
#     print(f"std={x.std()}")


def eval(x):

    # conv1 with scale=sqrt(act1.a2)
    x = torch.nn.functional.conv2d(x, weight=sd["conv1.weight"], bias=sd["conv1.bias"], stride=1, padding=1)
    a, b, c = herpn_eval("act1")
    a0, a1, a2 = a.reshape(1, -1, 1, 1), b.reshape(1, -1, 1, 1), c.reshape(1, -1, 1, 1)
    x = x * torch.sqrt(a2)

    # act1
    x = x * x + (a1 / torch.sqrt(a2)) * x + a0

    # pool1
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

    # conv2 with scale=sqrt(act2.a2)
    x = torch.nn.functional.conv2d(x, weight=sd["conv2.weight"], bias=sd["conv2.bias"], stride=1, padding=1)
    a, b, c = herpn_eval("act2")
    a0, a1, a2 = a.reshape(1, -1, 1, 1), b.reshape(1, -1, 1, 1), c.reshape(1, -1, 1, 1)
    x = x * torch.sqrt(a2)

    # act2
    x = x * x + (a1 / torch.sqrt(a2)) * x + a0

    # pool2
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

    # fc1 with scale=sqrt(act3.a2)
    x = x.view(x.size(0), -1)
    x = torch.nn.functional.linear(x, weight=sd["fc1.weight"], bias=sd["fc1.bias"])
    a, b, c = herpn_eval("act3")
    a0, a1, a2 = a.reshape(1, -1), b.reshape(1, -1), c.reshape(1, -1)
    x = x * torch.sqrt(a2)

    # act3
    x = x * x + (a1 / torch.sqrt(a2)) * x + a0

    # fc2
    x = torch.nn.functional.linear(x, weight=sd["fc2.weight"], bias=sd["fc2.bias"])

    return x


data, label = next(iter(test_loader))
y = eval(data)
label_pred = y.argmax(dim=1)

acc = (label_pred == label).float().mean()
print(f"Accuracy: {acc.item()*100:.2f}%")
