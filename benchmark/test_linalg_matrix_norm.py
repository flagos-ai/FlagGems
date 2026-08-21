import math

import pytest
import torch

import flag_gems

from . import base, consts, utils

VENDOR = flag_gems.vendor_name
_SVD_DTYPES = [torch.float32, torch.float64] if VENDOR == "nvidia" else [torch.float32]

SVD_SHAPES_SMALL = [
    (2, 64),
    (64, 3),
    (256, 8),
]

SVD_SHAPES_MEDIUM = [
    (32, 512),
    (64, 1024),
    (1024, 128),
    (64, 64),
]

SVD_SHAPES_LARGE = [
    (256, 1024),
    (1024, 512),
    (2048, 2),
    (512, 512),
    (512, 256),
]

# Batched SVD shapes — per-matrix (k, rows) within limits.
SVD_SHAPES_BATCHED = [
    (4, 32, 64),
    (8, 128, 256),
    (2, 128, 512),
    (16, 2, 16),
    (16, 2, 2048),
    (4, 4, 64, 64),
    (8, 4, 32, 32),
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

    # On Ascend: non-SVD ords (1, -1, inf, -inf, fro) crash/hang CANN native,
    # so skip them (no torch baseline to compare against).  Only benchmark
    # SVD ords (2, -2, 'nuc').
    if VENDOR == "ascend":
        if shape in _SVD_SHAPES and _svd_ords_allowed(shape, dtype):
            for ord_val in _SVD_ORDS:
                yield inp.clone(), ord_val
        return

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
        super().init_user_config()
        if flag_gems.vendor_name == "ascend":
            base.Config.mode = consts.BenchMode.OPERATOR

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        # Square-matrix sweep (fused-kernel ords, all levels).
        shapes += [
            (2, 128),
            (8, 8),
            (64, 64),
            (256, 256),
            (512, 512),
        ]
        # Batched shapes — always included (core + comprehensive).
        shapes += [
            (4, 32, 64),
            (8, 128, 256),
            (4, 4, 64, 64),
            (16, 2, 256),
        ]
        # SVD core shapes — always included, cover each dispatch path (float32-only).
        shapes += [
            (8, 1),
            (64, 2),
            (3, 4),
            (64, 3),
            (16, 64),
            (32, 128),
            (128, 64),
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
    # Ascend: SVD kernels are fp32 only (BiSheng rejects fp64; fp16/bf16
    # aren't supported by the Jacobi kernels).  Non-SVD ords are skipped
    # entirely (CANN native crashes).  Benchmark fp32 SVD ords only.
    if VENDOR == "ascend":
        bench_dtypes = [torch.float32]
    elif VENDOR == "iluvatar":
        bench_dtypes = consts.FLOAT_DTYPES
    else:
        bench_dtypes = consts.FLOAT_DTYPES + [torch.float64]

    bench = MatrixNormBenchmark(
        op_name="linalg_matrix_norm",
        input_fn=matrix_norm_input_fn,
        torch_op=torch.linalg.matrix_norm,
        gems_op=flag_gems.linalg_matrix_norm,
        dtypes=bench_dtypes,
    )
    bench.run()
