import torch

from .fused import *  # noqa: F403
from .ops import *  # noqa: F403

__version__ = "2.0"

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("abs", abs, "CUDA")
    lib.impl("add.Tensor", add, "CUDA")
    lib.impl("addmm", addmm, "CUDA")
    lib.impl("bitwise_and.Tensor", bitwise_and_tensor, "CUDA")
    lib.impl("bitwise_and.Scalar", bitwise_and_scalar, "CUDA")
    lib.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, "CUDA")
    lib.impl("bitwise_not", bitwise_not, "CUDA")
    lib.impl("bitwise_or.Tensor", bitwise_or_tensor, "CUDA")
    lib.impl("bitwise_or.Scalar", bitwise_or_scalar, "CUDA")
    lib.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, "CUDA")
    lib.impl("bmm", bmm, "CUDA")
    lib.impl("clamp", clamp, "CUDA")
    lib.impl("clamp.Tensor", clamp_tensor, "CUDA")
    lib.impl("cos", cos, "CUDA")
    lib.impl("cumsum", cumsum, "CUDA")
    lib.impl("div.Tensor", div, "CUDA")
    lib.impl("native_dropout", native_dropout, "AutogradCUDA")
    lib.impl("erf", erf, "CUDA")
    lib.impl("embedding", embedding, "AutogradCUDA")
    lib.impl("eq.Tensor", eq, "CUDA")
    lib.impl("eq.Scalar", eq_scalar, "CUDA")
    lib.impl("exp", exp, "CUDA")
    lib.impl("exponential_", exponential_, "CUDA")
    lib.impl("ge.Tensor", ge, "CUDA")
    lib.impl("ge.Scalar", ge_scalar, "CUDA")
    lib.impl("gelu", gelu, "CUDA")
    lib.impl("native_group_norm", group_norm, "AutogradCUDA")
    lib.impl("gt.Tensor", gt, "CUDA")
    lib.impl("gt.Scalar", gt_scalar, "CUDA")
    lib.impl("isfinite", isfinite, "CUDA")
    lib.impl("isinf", isinf, "CUDA")
    lib.impl("isnan", isnan, "CUDA")
    lib.impl("native_layer_norm", layer_norm, "AutogradCUDA")
    lib.impl("le.Tensor", le, "CUDA")
    lib.impl("le.Scalar", le_scalar, "CUDA")
    lib.impl("lt.Tensor", lt, "CUDA")
    lib.impl("lt.Scalar", lt_scalar, "CUDA")
    lib.impl("rms_norm", rms_norm, "CUDA")
    lib.impl("rand", rand, "CUDA")
    lib.impl("randn", randn, "CUDA")
    lib.impl("rand_like", rand_like, "CUDA")
    lib.impl("mean", mean, "CUDA")
    lib.impl("mean.dim", mean_dim, "CUDA")
    lib.impl("mm", mm, "CUDA")
    lib.impl("mul.Tensor", mul, "CUDA")
    lib.impl("mv", mv, "CUDA")
    lib.impl("ne.Tensor", ne, "CUDA")
    lib.impl("ne.Scalar", ne_scalar, "CUDA")
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
    lib.impl("where.self", where_self, "CUDA")
    lib.impl("where.ScalarSelf", where_scalar_self, "CUDA")
    lib.impl("where.ScalarOther", where_scalar_other, "CUDA")
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
    lib.impl("all", all, "CUDA")
    lib.impl("all.dim", all_dim, "CUDA")
    lib.impl("all.dims", all_dims, "CUDA")
    lib.impl("any", any, "CUDA")
    lib.impl("any.dim", any_dim, "CUDA")
    lib.impl("any.dims", any_dims, "CUDA")
    lib.impl("log_softmax.int", log_softmax, "AutogradCUDA")
    lib.impl("outer", outer, "AutogradCUDA")
    lib.impl("cross_entropy_loss", cross_entropy_loss, "AutogradCUDA")
    lib.impl("isclose", isclose, "CUDA")
    lib.impl("allclose", allclose, "CUDA")
    lib.impl("flip", flip, "CUDA")


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib


__all__ = [
    "enable",
    "use_gems",
]
