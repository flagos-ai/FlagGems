import torch
from .abs import abs
from .add import add
from .addmm import addmm
from .bmm import bmm
from .cumsum import cumsum
from .div import div
from .dropout import dropout
from .exp import exp
from .gelu import gelu
from .layernorm import layer_norm
from .mean import mean
from .mm import mm
from .mul import mul
from .pow import pow
from .reciprocal import reciprocal
from .relu import relu
from .rsqrt import rsqrt
from .silu import silu
from .sub import sub
from .triu import triu
from .softmax import softmax

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("abs", abs, "CUDA")
    lib.impl("abs.out", abs, "CUDA")
    lib.impl("add.Tensor", add, "CUDA")
    lib.impl("add.out", add, "CUDA")
    lib.impl("addmm", addmm, "CUDA")
    lib.impl("bmm", bmm, "CUDA")
    lib.impl("bmm.out", bmm, "CUDA")
    lib.impl("cumsum", cumsum, "CUDA")
    lib.impl("cumsum.out", cumsum, "CUDA")
    lib.impl("div.Tensor", div, "CUDA")
    lib.impl("div.out", div, "CUDA")
    lib.impl("dropout", dropout, "CUDA")
    lib.impl("exp", exp, "CUDA")
    lib.impl("exp.out", exp, "CUDA")
    lib.impl("gelu", gelu, "CUDA")
    lib.impl("layer_norm", layer_norm, "CUDA")
    lib.impl("mean", mean, "CUDA")
    lib.impl("mm", mm, "CUDA")
    lib.impl("mul.Tensor", mul, "CUDA")
    lib.impl("mul.out", mul, "CUDA")
    lib.impl("pow.Tensor_Scalar_out", pow, "CUDA")
    lib.impl("reciprocal", reciprocal, "CUDA")
    lib.impl("reciprocal.out", reciprocal, "CUDA")
    lib.impl("relu", relu, "CUDA")
    lib.impl("rsqrt", rsqrt, "CUDA")
    lib.impl("rsqrt.out", rsqrt, "CUDA")
    lib.impl("silu", silu, "CUDA")
    lib.impl("silu.out", silu, "CUDA")
    lib.impl("sub.Tensor", sub, "CUDA")
    lib.impl("sub.out", sub, "CUDA")
    lib.impl("triu", triu, "CUDA")
    lib.impl("triu.out", triu, "CUDA")
    lib.impl("softmax.int", softmax, "AutogradCUDA")


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib
