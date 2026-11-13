from .div import div_mode, div_mode_
from .hadamard_transform import hadamard_transform
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8

__all__ = [
    "div_mode",
    "div_mode_",
    "hadamard_transform",
    "matmul_bf16",
    "matmul_int8",
]

from .binary_add import binary_add
from .cat_copy import cat_copy
from .contiguous import contiguous
from .conv2d import conv2d
from .copy import copy
from .div import div
from .fill import fill
from .glu import glu

# 新增：Iluvatar FlagTree 优化算子
from .mm import addmm, bmm, mm
from .zeros import zeros

__all__ += [
    "mm",
    "addmm",
    "bmm",
    "conv2d",
    "glu",
    "binary_add",
    "cat_copy",
    "contiguous",
    "copy",
    "div",
    "fill",
    "zeros",
]

# 追加注册（原有代码不动，只加在文件末尾）
from flag_gems.runtime import backend_registry

from .ops import (
    addmm,
    binary_add,
    bmm,
    cat_copy,
    contiguous,
    conv2d,
    copy,
    div,
    fill,
    glu,
    mm,
    zeros,
)

backend_registry.register_impl("mm", mm, backend="iluvatar")
backend_registry.register_impl("addmm", addmm, backend="iluvatar")
backend_registry.register_impl("bmm", bmm, backend="iluvatar")
backend_registry.register_impl("conv2d", conv2d, backend="iluvatar")
backend_registry.register_impl("glu", glu, backend="iluvatar")
backend_registry.register_impl("binary_add", binary_add, backend="iluvatar")
backend_registry.register_impl("cat_copy", cat_copy, backend="iluvatar")
backend_registry.register_impl("contiguous", contiguous, backend="iluvatar")
backend_registry.register_impl("copy", copy, backend="iluvatar")
backend_registry.register_impl("div", div, backend="iluvatar")
backend_registry.register_impl("fill", fill, backend="iluvatar")
backend_registry.register_impl("zeros", zeros, backend="iluvatar")
