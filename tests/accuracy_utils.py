import importlib
import itertools
import random

import numpy as np
import torch

import flag_gems

from .conftest import QUICK_MODE, TO_CPU

fp64_is_supported = flag_gems.runtime.device.support_fp64
bf16_is_supported = flag_gems.runtime.device.support_bf16
int64_is_supported = flag_gems.runtime.device.support_int64


def TestForwardOnly():
    return flag_gems.vendor_name in []


def SkipVersion(module_name, skip_pattern):
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">"), f"Invalid comparison operator: {cmp}"
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from skip_pattern.")

    try:
        module = importlib.import_module(module_name)
        version = module.__version__
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

sizes_one = [1]
sizes_pow_2 = [2**d for d in range(4, 11, 2)]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = sizes_one + sizes_pow_2 + sizes_noalign
sizes_2d_nc = [1] if QUICK_MODE else [1, 16, 64, 512]
sizes_2d_nr = [1] if QUICK_MODE else [1, 5, 1024]

UT_SHAPES_1D = list((n,) for n in sizes_1d)
UT_SHAPES_2D = list(itertools.product(sizes_2d_nr, sizes_2d_nc))
POINTWISE_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(), (1,), (1024, 512), (20, 320, 15), (16, 128, 16, 4), (16, 7, 32, 8, 4)]
)
SPECIAL_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (1024, 512), (20, 320, 15), (16, 128, 16, 4), (16, 7, 32, 8, 4)]
)

FP8_QUANT_SHAPES = {
    "DTYPES": [torch.bfloat16],
    "NUM_TOKENS": [7] if QUICK_MODE else [7, 83, 2048],
    "D": [512] if QUICK_MODE else [512],
    "GROUP_SIZE": [512] if QUICK_MODE else [64, 128, 256, 512],
    "SEEDS": [0],
}

DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 512, 3)]
REDUCTION_SMALL_SHAPES = (
    [(1, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 512, 3)]
)
STACK_SHAPES = [
    [(16,), (16,)],
    [(16, 256), (16, 256)],
    [(20, 320, 15), (20, 320, 15), (20, 320, 15)],
]
CONTIGUOUS_SHAPE_STRIDES_1D = [
    ((1,), (1,)),
    ((1024,), (1,)),
    ((65535,), (1,)),
]
DILATED_SHAPE_STRIDES_1D = [
    ((1,), (2,)),
    ((1024,), (2,)),
    ((65535,), (2,)),
]
CONTIGUOUS_SHAPE_STRIDES_2D = [
    ((1, 512), (512, 1)),
    ((4096, 128), (128, 1)),
]
TRANSPOSED_SHAPE_STRIDES_2D = [
    ((512, 1), (1, 512)),
    ((128, 512), (1, 128)),
]
CONTIGUOUS_SHAPE_STRIDES_3D = [
    ((20, 320, 15), (4800, 15, 1)),
    ((200, 512, 3), (1536, 3, 1)),
]
TRANSPOSED_SHAPE_STRIDES_3D = [
    ((320, 20, 15), (15, 4800, 1)),
    ((3, 512, 32), (1, 3, 1536)),
]
SHAPE_STRIDES = (
    CONTIGUOUS_SHAPE_STRIDES_1D
    + DILATED_SHAPE_STRIDES_1D
    + CONTIGUOUS_SHAPE_STRIDES_2D
    + TRANSPOSED_SHAPE_STRIDES_2D
    + CONTIGUOUS_SHAPE_STRIDES_3D
    + TRANSPOSED_SHAPE_STRIDES_3D
)

IRREGULAR_SHAPE_STRIDES = [((10, 10, 10, 8, 4), (1, 3200, 23, 320, 80))]

UPSAMPLE_SHAPES = [
    (32, 16, 16, 4),
    (15, 37, 16, 4),
    (3, 5, 16, 4),
    (128, 192, 16, 4),
    (3, 7, 16, 4),
]

# 1D upsample uses (N, C, W) shapes derived from the 2D cases above.
UPSAMPLE_SHAPES_1D = [s[:3] for s in UPSAMPLE_SHAPES]

UPSAMPLE_SHAPES_3D = [
    (4, 8, 32, 8, 4),
    (3, 5, 17, 8, 4),
    (2, 16, 8, 8, 4),
    (12, 24, 16, 8, 4),
    (1, 2, 32, 8, 4),
]

SWIGLU_SPECIAL_SHAPES = (
    [(2, 19, 8)]
    if QUICK_MODE
    else [
        (2,),
        (64,),
        (32, 64),
        (256, 512),
        (1, 128),
        (8, 16, 32),
        (16, 32, 32),
        (20, 320, 16),
        (4, 8, 16, 4),
        (8, 16, 16, 4),
        (10,),
        (20, 30),
    ]
)

KRON_SHAPES = [
    [(), (2, 3)],
    [(2, 3), ()],
    [(0, 3), (2, 3)],
    [(2, 3), (0,)],
    [(0,), (0,)],
    [(), ()],
    [(1,), (2,)],
    [(2,), (3,)],
    [(2, 2), (3, 3)],
    [(1, 2, 3), (2, 3, 4)],
    [(1,), (2, 2)],
    [(1, 2), (3, 4, 5)],
    [(2,), (3, 4, 5, 4)],
    [(2, 3, 4), (1,)],
    [(5, 5), (4, 4)],
    [(3, 3, 3), (2, 2, 2)],
    [(4, 4, 4, 4), (2, 2, 2, 2)],
    [(2, 3, 4), (3, 4, 5)],
    [(1, 3, 5), (2, 4, 6)],
    [(2, 4, 6, 4), (1, 3, 5, 4)],
    [(1, 3), (1, 4)],
    [(1, 1, 3), (1, 1, 2)],
    [(2, 1, 4), (3, 1, 5)],
    [(2, 2, 2, 2, 2), (1, 1, 1, 1, 1)],
    [(1, 2, 3, 4, 4), (2, 3, 4, 5, 4)],
    [(1,), (1,)],
    [(10,), (10,)],
    [(2, 3), (3, 2)],
    [(3, 3), (3, 3)],
    [(1, 1, 1), (2, 2, 2)],
]
# Add some test cases with zeor-dimensional tensor and zero-sized tensors.
PRIMARY_FLOAT_DTYPES = [torch.float16, torch.float32]
FLOAT_DTYPES = (
    PRIMARY_FLOAT_DTYPES + [torch.bfloat16]
    if bf16_is_supported
    else PRIMARY_FLOAT_DTYPES
)

ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64] if fp64_is_supported else FLOAT_DTYPES
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = INT_DTYPES + [torch.int64] if int64_is_supported else INT_DTYPES
BOOL_TYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex32, torch.complex64]

SCALARS = [0.001, -0.999, 100.001, -111.999]
STACK_DIM_LIST = [-2, -1, 0, 1]

ARANGE_START = [0] if TO_CPU else [0, 1, 3]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def to_cpu(res, ref):
    if TO_CPU and isinstance(res, torch.Tensor) and isinstance(ref, torch.Tensor):
        res = res.to("cpu")
        assert ref.device == torch.device("cpu")
    return res


def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=1e-4):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim, atol=atol
    )


def gems_assert_equal(res, ref, equal_nan=False):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_equal(res, ref, equal_nan=equal_nan)


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1,)
    return t


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
