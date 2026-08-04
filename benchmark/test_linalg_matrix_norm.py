import math

import pytest
import torch

import flag_gems

from . import base, consts, utils

VENDOR = flag_gems.vendor_name
_SVD_DTYPES = [torch.float32, torch.float64] if VENDOR == "nvidia" else [torch.float32]

SVD_SHAPES_SMALL = [
    (2, 64),
    (64, 2),  # k=2 (rank2 closed form)
    (2, 256),
    (256, 2),  # k=2 near rows limit
    (4, 64),
    (64, 4),  # k=4 (Jacobi minimum)
    (8, 256),
    (256, 8),  # k=8
    (16, 256),
    (256, 16),  # k=16
    (4, 4),
    (8, 8),
    (16, 16),  # square
]

SVD_SHAPES_MEDIUM = [
    (32, 512),
    (512, 32),  # k=32
    (64, 1024),
    (1024, 64),  # k=64
    (128, 1024),
    (1024, 128),  # k=128
    (32, 32),
    (64, 64),
    (128, 128),
]

SVD_SHAPES_LARGE = [
    (256, 1024),
    (1024, 256),  # k=256
    (384, 1024),
    (1024, 384),  # k=384
    (512, 1024),
    (1024, 512),  # k=512
    (2, 2048),
    (2048, 2),  # k=2, rows=2048
    (256, 256),
    (512, 512),
]

# Batched SVD shapes — per-matrix (k, rows) within limits.
SVD_SHAPES_BATCHED = [
    (4, 32, 64),
    (8, 64, 128),
    (2, 128, 512),
    (16, 2, 256),
]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

# Union of all SVD-specific shapes — only these shapes get ord=2/-2/nuc.
_SVD_ORDS = (2, -2, "nuc")
_SVD_SHAPES = set(
    SVD_SHAPES_SMALL + SVD_SHAPES_MEDIUM + SVD_SHAPES_LARGE + SVD_SHAPES_BATCHED
)


def _svd_ords_allowed(shape, dtype):
    """Return True when SVD-based ords can be tested for this shape/dtype."""
    if dtype not in (torch.float32, torch.float64):
        return False
    k = min(shape[-2], shape[-1])
    rows = max(shape[-2], shape[-1])
    if k > 512 or rows > 2048:
        return False
    return True


def matrix_norm_input_fn(shape, dtype, device):
    """Yield (input, ord) or (input, ord, dim) tuples for each supported ord."""
    # generate_tensor_input only handles float16/32/bf16; handle float64 directly.
    if dtype == torch.float64:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = utils.generate_tensor_input(shape, dtype, device)

    for ord_val in (1, -1, float("inf"), float("-inf")):
        yield inp.clone(), ord_val
    yield inp.clone(), "fro"

    # SVD-based norms for shapes in the SVD tier lists.
    if shape in _SVD_SHAPES and _svd_ords_allowed(shape, dtype):
        for ord_val in _SVD_ORDS:
            yield inp.clone(), ord_val

    if len(shape) > 2:
        yield inp.clone(), "fro", (-2, -1)


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MatrixNormBenchmark(base.GenericBenchmark2DOnly):
    # Maximum total elements to keep benchmark time reasonable.
    MAX_ELEMENTS = 128 * 1024 * 1024  # 128M — include all default shapes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = [
            s for s in self.shapes if len(s) >= 2 and math.prod(s) <= self.MAX_ELEMENTS
        ]

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        # Square-matrix sweep (fused-kernel ords, all levels).
        shapes += [
            (2, 128),
            (128, 2),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]
        # Batched shapes — always included (core + comprehensive).
        shapes += [
            (4, 32, 64),
            (8, 64, 128),  # single batch dim, small k
            (8, 128, 256),
            (4, 4, 64, 64),  # batch + multi-batch
            (16, 2, 256),  # rank-2 batched
        ]
        # SVD core shapes — always included, cover each dispatch path (float32-only).
        shapes += [
            (2, 64),
            (64, 2),  # k=2: rank2 closed form
            (4, 64),
            (64, 4),  # k=4: Jacobi minimum (15 sweeps)
            (16, 64),
            (64, 16),  # k=16: small Jacobi (15 sweeps)
            (32, 128),
            (128, 32),  # k=32: medium Jacobi (15 sweeps)
            (128, 64),  # k=64: large Jacobi (20 sweeps)
        ]
        # SVD tiers + batched SVD — comprehensive only (slow, many kernel launches).
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            shapes += SVD_SHAPES_SMALL
            shapes += SVD_SHAPES_MEDIUM
            shapes += SVD_SHAPES_LARGE
            shapes += SVD_SHAPES_BATCHED
        return [s for s in shapes if math.prod(s) <= self.MAX_ELEMENTS]


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.linalg_matrix_norm
def test_linalg_matrix_norm():
    # Wrapper: C++ op requires explicit dim; fill in defaults to match ATen API.
    if hasattr(torch.ops.flag_gems, "linalg_matrix_norm"):
        _cpp_op = torch.ops.flag_gems.linalg_matrix_norm

        def _gems_op(*args, **kwargs):
            A, ord_val = args[0], args[1]
            dim = args[2] if len(args) > 2 else (-2, -1)
            keepdim = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
            if isinstance(ord_val, str):
                return _cpp_op.str_ord(A, ord_val, dim, keepdim)
            return _cpp_op(A, float(ord_val), dim, keepdim)

        gems_op = _gems_op
    else:
        gems_op = flag_gems.linalg_matrix_norm

    bench = MatrixNormBenchmark(
        op_name="linalg_matrix_norm",
        input_fn=matrix_norm_input_fn,
        torch_op=torch.ops.aten.linalg_matrix_norm,
        gems_op=gems_op,
        dtypes=consts.FLOAT_DTYPES
        + ([torch.float64] if VENDOR == "nvidia" else []),  # float64 SVD only on NVIDIA
    )
    bench.run()
