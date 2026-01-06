from backend_utils import Autograd

from . import add, gelu, cat

from .cat import cat
from .add import add, add_
from .abs import abs, abs_
from .cos import cos, cos_
from .tanh import tanh, tanh_, tanh_backward
from .eq import eq, eq_scalar
from .exp import exp, exp_
from .mul import mul, mul_
from .sub import sub, sub_
from .neg import neg, neg_
from .ge import ge, ge_scalar
from .gt import gt, gt_scalar
from .le import le, le_scalar
from .lt import lt, lt_scalar
from .logical_and import logical_and
from .logical_not import logical_not
from .isinf import isinf
from .isnan import isnan
from .ones import ones
from .ones_like import ones_like
from .fill import fill_scalar, fill_scalar_, fill_tensor, fill_tensor_
from .flip import flip
from .full import full
from .full_like import full_like
from .zeros import zeros
from .zeros_like import zeros_like
from .maximum import maximum
from .minimum import minimum
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_not import bitwise_not, bitwise_not_
from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
# from .all import all_dim, all_dims, all

from .outer import outer

from .gelu_and_mul import gelu_and_mul


def get_specific_ops():
    return (
        ("add.Tensor", add.add, Autograd.disable),
        ("gelu", gelu.gelu, Autograd.enable),
        # ("randn", randn.ts_randn, Autograd.enable),
    )


def get_unused_ops():
    return ("cumsum", "cos")


# __all__ = ["get_specific_ops", "get_unused_ops", "cat", "all_dims", "all_dim", "all", outer]

__all__ = ["get_specific_ops", "get_unused_ops", "cat", "outer", "add", "add_",
           "abs", "abs_",
           "bitwise_and_scalar", "bitwise_and_scalar_", "bitwise_and_scalar_tensor", "bitwise_and_tensor", "bitwise_and_tensor_",
           "bitwise_or_scalar", "bitwise_or_scalar_", "bitwise_or_scalar_tensor", "bitwise_or_tensor", "bitwise_or_tensor_",
           "cos", "cos_",
           "eq", "eq_scalar",
           "exp", "exp_",
           "mul", "mul_",
           "neg", "neg_",
           "sub", "sub_",
           "tanh", "tanh_", "tanh_backward",
           #"all_dims", "all_dim", "all",
           "ge", "ge_scalar",
           "gt", "gt_scalar",
           "le", "le_scalar",
           "lt", "lt_scalar",
           "isinf",
           "isnan",
           "ones", "ones_like",
           "fill_scalar", "fill_tensor", "fill_scalar_", "fill_tensor_",
           "flip",
           "full", "full_like",
           "zeros", "zeros_like",
           "gelu_and_mul",
           "maximum", "minimum",
           "logical_and",
           "logical_not"]
