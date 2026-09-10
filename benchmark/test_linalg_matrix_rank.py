import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device

from . import base, consts

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"
IS_ILUVATAR = VENDOR_NAME == "iluvatar"
IS_THEAD = VENDOR_NAME == "thead"

# Skip float64 where the device lacks native FP64 (capability bit, e.g.
# Ascend -- whose aclnn svd_npu is float32-only -- or Iluvatar without an
# FP64 solver): neither our kernels nor the torch native baseline can run
# there.  float64 accuracy is still exercised in tests/ against the CPU
# reference.
_SUPPORT_FP64 = getattr(runtime_device, "support_fp64", True)
MATRIX_RANK_DTYPES = (
    [torch.float32, torch.float64] if _SUPPORT_FP64 else [torch.float32]
)


MATRIX_RANK_CORE_SHAPES = [
    (1, 256),
    (256, 1),
    (2, 256),
    (256, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (33, 33),
    (64, 64),
]

MATRIX_RANK_COMPREHENSIVE_SHAPES = [
    # Tall and wide matrices exercise both workspace orientations.
    (8, 256),
    (256, 8),
    (16, 512),
    (512, 16),
    (32, 1024),
    (1024, 32),
    (64, 512),
    (512, 64),
    # Default-dispatch band (QR -> exact paths at k = 256): 65..255 stay on
    # unpivoted QR, 256+ on the exact bidiagonalization/tridiagonalization.
    # 65..192 are sampled densely because the exact paths have tile-edge
    # performance dips there (which is why the boundary sits at 256).
    (65, 65),
    (80, 80),
    (96, 96),
    (112, 112),
    (120, 120),
    (127, 127),
    # Medium, large, and current native-support boundaries.
    (128, 128),
    (129, 129),
    (160, 160),
    (192, 192),
    (256, 256),
    (512, 512),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
    # Non-square shapes straddling the middle of the QR band (k = 129 is the
    # long-row / mid-k worst case: the QR panel and the bidiagonalization
    # both pay O(rows) per step).
    (129, 512),
    (512, 129),
    (129, 2048),
    (2048, 129),
    # Single- and multi-dimensional batches of small/medium matrices.
    (32, 8, 8),
    (8, 16, 16),
    (4, 32, 32),
    (2, 64, 64),
    (8, 64, 16),
    (8, 16, 64),
    (2, 4, 16, 16),
    (2, 129, 129),
    (8, 129, 129),
    (2, 256, 256),
]

MATRIX_RANK_HERMITIAN_CORE_SHAPES = [
    (1, 1),
    (2, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (33, 33),
    (64, 64),
]

MATRIX_RANK_HERMITIAN_COMPREHENSIVE_SHAPES = [
    # herm 65..255 uses unpivoted QR by default and 256+ the one-sided
    # tridiagonalization (since the stage-8 dispatch switch); 65/129/257
    # sample both sides of the boundary.
    (65, 65),
    (128, 128),
    (129, 129),
    (256, 256),
    (257, 257),
    (512, 512),
    (1024, 1024),
    (32, 8, 8),
    (8, 16, 16),
    (4, 32, 32),
    (2, 64, 64),
    (2, 4, 16, 16),
]


def _select_shapes(core_shapes, comprehensive_shapes):
    shapes = core_shapes.copy()
    if (
        not base.Config.query
        and base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
    ):
        shapes.extend(comprehensive_shapes)
    return shapes


class MatrixRankBenchmark(base.GenericBenchmark):
    """Benchmark for torch.linalg.matrix_rank."""

    DEFAULT_SHAPE_DESC = "*, M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = _select_shapes(
            MATRIX_RANK_CORE_SHAPES,
            MATRIX_RANK_COMPREHENSIVE_SHAPES,
        )


class MatrixRankHermitianBenchmark(base.GenericBenchmark):
    """Benchmark for torch.linalg.matrix_rank with hermitian=True."""

    DEFAULT_SHAPE_DESC = "*, N, N"

    def set_shapes(self, shape_file_path=None):
        shapes = _select_shapes(
            MATRIX_RANK_HERMITIAN_CORE_SHAPES,
            MATRIX_RANK_HERMITIAN_COMPREHENSIVE_SHAPES,
        )
        if IS_THEAD:
            # HGGC lacks cusolverDnXsyevBatched_bufferSize, so native Torch
            # cannot provide a device-side baseline for Hermitian batches.
            # Skip those shapes instead of reporting CPU/composed speedups.
            shapes = [shape for shape in shapes if len(shape) == 2]
        if IS_ILUVATAR:
            # Iluvatar's native Hermitian path works for the remaining core,
            # comprehensive and batched shapes, but the (1, 1) benchmark
            # hangs before completing its native baseline.
            shapes = [shape for shape in shapes if shape != (1, 1)]
        self.shapes = shapes


class MatrixRankOverloadBenchmark(base.GenericBenchmark):
    """Core-shape benchmark for matrix_rank overload adapters."""

    DEFAULT_SHAPE_DESC = "*, M, N"

    def set_shapes(self, shape_file_path=None):
        # Overloads share the same decomposition kernels, so core shapes are
        # sufficient to measure tolerance conversion and out-copy overhead.
        self.shapes = MATRIX_RANK_CORE_SHAPES.copy()


def _composed_matrix_rank(matrix, atol=None, rtol=None, hermitian=False):
    """matrix_rank composed from native NPU ops (svdvals + reduction).

    torch.linalg.matrix_rank(hermitian=True) dispatches to
    aten::_linalg_eigh.eigenvalues, which has no NPU implementation and falls
    back to the CPU. This composed version keeps the reference latency on the
    NPU: singular values of a Hermitian matrix are the absolute eigenvalues,
    so svdvals (native aclnn) plus a threshold count reproduces the semantics.
    """
    if hermitian:
        # eigh reads only the lower triangle; mirror that before the SVD.
        matrix = torch.tril(matrix) + torch.tril(matrix, -1).mT
    svals = torch.linalg.svdvals(matrix)
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = max(matrix.shape[-2], matrix.shape[-1]) * torch.finfo(matrix.dtype).eps
    smax = svals.amax(dim=-1, keepdim=True)
    tol = torch.clamp_min(smax * rtol, atol)
    return (svals > tol).sum(dim=-1)


@pytest.mark.linalg_matrix_rank_atol_rtol_float
def test_linalg_matrix_rank():
    def matrix_rank_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        yield (matrix,)

    bench = MatrixRankBenchmark(
        input_fn=matrix_rank_input_fn,
        op_name="linalg_matrix_rank",
        torch_op=torch.linalg.matrix_rank,
        dtypes=MATRIX_RANK_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()


@pytest.mark.linalg_matrix_rank_atol_rtol_float
def test_linalg_matrix_rank_hermitian():
    def matrix_rank_hermitian_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        matrix = matrix + matrix.mT
        yield matrix, {"hermitian": True}

    if IS_ASCEND:
        # torch.linalg.matrix_rank(hermitian=True) falls back to the CPU on
        # Ascend (aten::_linalg_eigh.eigenvalues has no NPU kernel). Use a
        # baseline composed of native NPU ops instead, mirroring the
        # cholesky_solve benchmark.
        torch_op = _composed_matrix_rank
    else:
        torch_op = torch.linalg.matrix_rank

    bench = MatrixRankHermitianBenchmark(
        input_fn=matrix_rank_hermitian_input_fn,
        op_name="linalg_matrix_rank_hermitian",
        torch_op=torch_op,
        dtypes=MATRIX_RANK_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_matrix_rank)
    bench.run()


@pytest.mark.parametrize(
    "op_name,legacy,tensor_tolerance,use_out,gems_op",
    [
        pytest.param(
            "linalg_matrix_rank",
            True,
            False,
            False,
            flag_gems.linalg_matrix_rank_tol,
            marks=pytest.mark.linalg_matrix_rank,
            id="legacy-float",
        ),
        pytest.param(
            "linalg_matrix_rank_tol_tensor",
            True,
            True,
            False,
            flag_gems.linalg_matrix_rank_tol,
            marks=pytest.mark.linalg_matrix_rank_tol_tensor,
            id="legacy-tensor",
        ),
        pytest.param(
            "linalg_matrix_rank_out",
            True,
            False,
            True,
            flag_gems.linalg_matrix_rank_tol_out,
            marks=pytest.mark.linalg_matrix_rank_out,
            id="legacy-float-out",
        ),
        pytest.param(
            "linalg_matrix_rank_out_tol_tensor",
            True,
            True,
            True,
            flag_gems.linalg_matrix_rank_tol_out,
            marks=pytest.mark.linalg_matrix_rank_out_tol_tensor,
            id="legacy-tensor-out",
        ),
        pytest.param(
            "linalg_matrix_rank_atol_rtol_tensor",
            False,
            True,
            False,
            flag_gems.linalg_matrix_rank,
            marks=pytest.mark.linalg_matrix_rank_atol_rtol_tensor,
            id="atol-rtol-tensor",
        ),
        pytest.param(
            "linalg_matrix_rank_atol_rtol_float_out",
            False,
            False,
            True,
            flag_gems.linalg_matrix_rank_out,
            marks=pytest.mark.linalg_matrix_rank_atol_rtol_float_out,
            id="atol-rtol-float-out",
        ),
        pytest.param(
            "linalg_matrix_rank_atol_rtol_tensor_out",
            False,
            True,
            True,
            flag_gems.linalg_matrix_rank_out,
            marks=pytest.mark.linalg_matrix_rank_atol_rtol_tensor_out,
            id="atol-rtol-tensor-out",
        ),
    ],
)
def test_linalg_matrix_rank_overloads(
    op_name,
    legacy,
    tensor_tolerance,
    use_out,
    gems_op,
):
    def matrix_rank_overload_input_fn(shape, cur_dtype, device):
        matrix = torch.randn(shape, dtype=cur_dtype, device=device)
        tolerance = (
            torch.tensor(0.0, dtype=cur_dtype, device=device)
            if tensor_tolerance
            else 0.0
        )
        args = [matrix]
        kwargs = {}
        if legacy:
            args.append(tolerance)
        else:
            kwargs["atol"] = tolerance
        if use_out:
            kwargs["out"] = torch.empty(shape[:-2], dtype=torch.int64, device=device)
        if kwargs:
            args.append(kwargs)
        yield tuple(args)

    bench = MatrixRankOverloadBenchmark(
        input_fn=matrix_rank_overload_input_fn,
        op_name=op_name,
        torch_op=torch.linalg.matrix_rank,
        dtypes=MATRIX_RANK_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()
