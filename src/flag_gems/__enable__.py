import torch
from .abs import abs
from .add import add
from .addmm import addmm
from .bitwise_not import bitwise_not
from .bmm import bmm
from .cos import cos
from .cumsum import cumsum
from .div import div
from .dropout import native_dropout
from .exp import exp
from .gelu import gelu
from .groupnorm import group_norm
from .isinf import isinf
from .isnan import isnan
from .layernorm import layer_norm
from .mean import mean, mean_dim
from .mm import mm
from .mul import mul
from .neg import neg
from .pow_scalar import pow_scalar
from .pow_tensor_scalar import pow_tensor_scalar
from .pow_tensor_tensor import pow_tensor_tensor
from .reciprocal import reciprocal
from .relu import relu
from .rsqrt import rsqrt
from .sigmoid import sigmoid
from .silu import silu
from .sin import sin
from .softmax import softmax
from .sub import sub
from .tanh import tanh
from .triu import triu
from .var_mean import var_mean
from .vector_norm import vector_norm

from .max import max, max_dim
from .min import min, min_dim
from .amax import amax
from .argmax import argmax
from .prod import prod, prod_dim
from .sum import sum, sum_dim


aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("abs", abs, "CUDA")
    lib.impl("add.Tensor", add, "CUDA")
    lib.impl("addmm", addmm, "CUDA")
    lib.impl("bitwise_not", bitwise_not, "CUDA")
    lib.impl("bmm", bmm, "CUDA")
    lib.impl("cos", cos, "CUDA")
    lib.impl("cumsum", cumsum, "CUDA")
    lib.impl("div.Tensor", div, "CUDA")
    lib.impl("native_dropout", native_dropout, "AutogradCUDA")
    lib.impl("exp", exp, "CUDA")
    lib.impl("gelu", gelu, "CUDA")
    lib.impl("native_group_norm", group_norm, "AutogradCUDA")
    lib.impl("isinf", isinf, "CUDA")
    lib.impl("isnan", isnan, "CUDA")
    lib.impl("native_layer_norm", layer_norm, "AutogradCUDA")
    lib.impl("mean", mean, "CUDA")
    lib.impl("mean.dim", mean_dim, "CUDA")
    lib.impl("mm", mm, "CUDA")
    lib.impl("mul.Tensor", mul, "CUDA")
    lib.impl("neg", neg, "CUDA")
    lib.impl("pow.Scalar", pow_scalar, "CUDA")
    lib.impl("pow.Tensor_Scalar", pow_tensor_scalar, "CUDA")
    lib.impl("pow.Tensor_Tensor", pow_tensor_tensor, "CUDA")
    lib.impl("reciprocal", reciprocal, "CUDA")
    lib.impl("relu", relu, "AutogradCUDA")
    lib.impl("rsqrt", rsqrt, "CUDA")
    lib.impl("sigmoid", sigmoid, "AutogradCUDA")
    lib.impl("silu", silu, "AutogradCUDA")
    lib.impl("sin", sin, "CUDA")
    lib.impl("softmax.int", softmax, "AutogradCUDA")
    lib.impl("sub.Tensor", sub, "CUDA")
    lib.impl("tanh", tanh, "AutogradCUDA")
    lib.impl("triu", triu, "CUDA")
    lib.impl("var_mean.correction", var_mean, "CUDA")
    lib.impl("linalg_vector_norm", vector_norm, "CUDA")

    lib.impl("max", max, "CUDA")
    lib.impl("max.dim", max_dim, "CUDA")
    lib.impl("min", min, "CUDA")
    lib.impl("min.dim", min_dim, "CUDA")
    lib.impl("amax", amax, "CUDA")
    lib.impl("argmax", argmax, "CUDA")
    lib.impl("prod", prod, "CUDA")
    lib.impl("prod.dim_int", prod_dim, "CUDA")
    lib.impl("sum", sum, "CUDA")
    lib.impl("sum.dim_IntList", sum_dim, "CUDA")


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib
