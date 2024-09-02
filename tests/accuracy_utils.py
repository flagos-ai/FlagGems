import itertools

import torch

from .conftest import TO_CPU

major, minor = torch.__version__.split(".")[:2]
skip_expr = major < "2" or minor < "2"
skip_reason = "PyTorch < 2.2.0 does not support"


RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

sizes_one = [1]
sizes_pow_2 = [2**d for d in range(4, 11, 2)]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = sizes_one + sizes_pow_2 + sizes_noalign
sizes_2d_nc = [1, 16, 64, 1000]
sizes_2d_nr = [1, 5, 1024]

UT_SHAPES_1D = list((n,) for n in sizes_1d)
UT_SHAPES_2D = list(itertools.product(sizes_2d_nr, sizes_2d_nc))
POINTWISE_SHAPES = [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)]
DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(4096, 256 * i) for i in range(1, 10, 2)]
MNK_SHAPES = [15, 160, 1024]
REDUCTION_MNK_SHAPES = [(15, 160, 1024), (16, 1025, 255)]
ONE_DIM_SHAPES = [(256 * i + 7,) for i in range(1, 10, 2)]

DIM_POINTWISE_SHAPES = [
    (1024, 1024, 1),
    (16, 1024, 256),
    (16, 7, 128, 64, 64),
    (20, 320, 15),
]
DIMS = [[0], [-2], [2], [0, 2], [2, 1], [0, -1, 1]]
TILE_DIMS = [(2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
ALL_FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = [torch.int16, torch.int32, torch.int64]

SCALARS = [0.001, -0.999, 100.001, -111.999]
DIM_LIST = [0, 1]
DIMS_LIST = [0, 1, [0, 1], [1, 0]]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def gems_assert_close(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_equal(a, b):
    if TO_CPU:
        a = a.to("cpu")
    assert torch.equal(a, b)
