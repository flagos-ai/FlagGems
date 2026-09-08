from collections import namedtuple

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if VENDOR == "nvidia":
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]

# pivot=False is only supported on CUDA
if DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]

_CHECK_ERRORS_VALUES = [False, True]

if VENDOR == "ascend":
    Config.mode = consts.BenchMode.OPERATOR

# Use the same shapes as linalg_lu_factor for consistency
LU_FACTOR_EX_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 512),
    (32, 16),
    (16, 32),
    (128, 64),
    (64, 128),
    (4, 32, 32),
    (128, 16, 16),
    (1024, 512, 512),
]

LinalgLUFactorExResult = namedtuple("LinalgLUFactorExResult", ["LU", "pivots", "info"])


# ---------------------------------------------------------------------------
# Manual Python reference implementation for Ascend
# (matches the pattern in benchmark/test_linalg_lu_factor.py)
# ---------------------------------------------------------------------------


def _swap_rows(lu, i, pivot_row):
    # In-place gather/scatter swap: only (*batch_shape, n) temporaries instead
    # of the full-matrix broadcast temporaries of the mask-based swap.  Keeping
    # lu in place also means the pivot values recorded in pivot_vals keep
    # pointing at the one live lu tensor instead of pinning a fresh 1GB copy
    # per iteration (which OOMs the device for batched shapes like
    # (1024, 512, 512)).  Note: torch.gather/scatter_ are used instead of
    # advanced indexing (lu[..., pivot_row, :]) because torch_npu expands
    # tensor indices against the batch dims, yielding (b, b, n) instead of
    # (b, n).
    batch_shape = lu.shape[:-2]
    n = lu.shape[-1]
    row_i = lu[..., i, :].clone()
    idx = pivot_row.view(*batch_shape, 1, 1).expand(*batch_shape, 1, n)
    row_p = lu.gather(dim=-2, index=idx).squeeze(-2)
    lu[..., i, :] = row_p
    lu.scatter_(dim=-2, index=idx, src=row_i.unsqueeze(-2))
    return lu


def _lu_factor_pivot_ex(lu, m, n, k):
    """LU factorization with partial pivoting + info tracking."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)
    # Collect pivots in a plain Python list (zero extra device ops per step)
    # and compute info once after the loop with vector ops.
    pivot_vals = []

    for i in range(k):
        col = lu[..., i:, i].abs()
        pivot_rel = torch.argmax(col, dim=-1)
        pivot_row = pivot_rel + i
        pivots[..., i] = (pivot_row + 1).to(torch.int32)

        lu = _swap_rows(lu, i, pivot_row)

        # .clone() so the recorded value does not pin lu's storage (a plain
        # view would keep the referenced lu storage alive; combined with a
        # rebinding lu this OOMs the device for batched shapes).
        pivot_val = lu[..., i, i].clone()
        pivot_vals.append(pivot_val)

        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    # First zero/NaN pivot position (1-indexed), 0 if none.
    diag = torch.stack(pivot_vals, dim=-1)  # (*batch_shape, k)
    bad_mask = (diag == 0) | torch.isnan(diag)
    has_bad = torch.any(bad_mask, dim=-1)
    first_bad = torch.argmax(bad_mask.to(torch.int32), dim=-1)
    info = torch.where(has_bad, first_bad + 1, 0).to(torch.int32)

    return lu, pivots, info


def _lu_factor_no_pivot_ex(lu, m, n, k):
    """LU factorization without pivoting + info tracking."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)
    pivot_vals = []

    for i in range(k):
        pivots[..., i] = i + 1
        # .clone() so the recorded value does not pin lu's storage (a plain
        # view would keep the referenced lu storage alive; combined with a
        # rebinding lu this OOMs the device for batched shapes).
        pivot_val = lu[..., i, i].clone()
        pivot_vals.append(pivot_val)

        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    # First zero/NaN pivot position (1-indexed), 0 if none.
    diag = torch.stack(pivot_vals, dim=-1)  # (*batch_shape, k)
    bad_mask = (diag == 0) | torch.isnan(diag)
    has_bad = torch.any(bad_mask, dim=-1)
    first_bad = torch.argmax(bad_mask.to(torch.int32), dim=-1)
    info = torch.where(has_bad, first_bad + 1, 0).to(torch.int32)

    return lu, pivots, info


def ops_lu_factor_ex(input, *, pivot=True):
    """Manual Python reference for linalg_lu_factor_ex."""
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor_ex: Expected input to have at least 2 dimensions"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError("Only float32 is supported")
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError("Empty matrices are not supported")
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")

    input_contiguous = input.contiguous()
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    lu = input_contiguous.clone()

    if pivot:
        lu, pivots, info = _lu_factor_pivot_ex(lu, m, n, k)
    else:
        lu, pivots, info = _lu_factor_no_pivot_ex(lu, m, n, k)

    return LinalgLUFactorExResult(lu, pivots, info)


if VENDOR == "ascend":

    def _torch_lu_factor_ex(inp, *, pivot=True, check_errors=False, out=None):
        """Vendor-aware wrapper: uses manual ops on Ascend, native on other vendors.

        Runs the Python reference directly on the NPU input, the same way the
        linalg_lu_factor benchmark reference does.  With the optimized composite
        (info computed once after the loop, no per-iteration extra ops) a full
        [256, 256] call costs ~200ms on NPU.  Moving the input to CPU first is
        much worse: on many-core hosts torch's intra-op thread pool makes each
        tiny torch CPU op cost several ms (measured ~13ms per iteration with
        192 threads), i.e. ~4-20s per call.
        """
        res = ops_lu_factor_ex(inp, pivot=pivot)
        if check_errors:
            info_val = res.info
            failed = info_val != 0
            if torch.any(failed).item():
                info_cpu = info_val.detach().cpu().reshape(-1)
                first_info = int(info_cpu[info_cpu != 0][0].item())
                raise RuntimeError(
                    "torch.linalg.lu_factor_ex: U[{},{}] is zero".format(
                        first_info, first_info
                    )
                )
        if out is not None:
            lu_out, piv_out, info_out = out
            lu_out.copy_(res.LU)
            piv_out.copy_(res.pivots)
            info_out.copy_(res.info)
            return lu_out, piv_out, info_out
        return res.LU, res.pivots, res.info

else:
    _torch_lu_factor_ex = torch.linalg.lu_factor_ex


class LinalgLuFactorExBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot, check_errors"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_FACTOR_EX_SHAPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                # Only test check_errors variations for pivot=True to keep
                # the benchmark size manageable.
                check_errors_list = _CHECK_ERRORS_VALUES if pivot else [False]
                for check_errors in check_errors_list:
                    inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                    yield inp, {"pivot": pivot, "check_errors": check_errors}


@pytest.mark.linalg_lu_factor_ex
def test_linalg_lu_factor_ex():
    bench = LinalgLuFactorExBenchmark(
        op_name="linalg_lu_factor_ex",
        torch_op=_torch_lu_factor_ex,
        gems_op=flag_gems.linalg_lu_factor_ex,
        dtypes=_TEST_DTYPES,
    )
    bench.run()


class LinalgLuFactorExOutBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot, check_errors"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_FACTOR_EX_SHAPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                if pivot and VENDOR != "ascend":
                    # out variant for pivot=True already covered by main benchmark
                    continue
                k = min(inp_shape[-2], inp_shape[-1])
                batch_shape = inp_shape[:-2]
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                LU = torch.empty(inp.shape, dtype=dtype, device=inp.device)
                pivots = torch.empty(
                    (*batch_shape, k), dtype=torch.int32, device=inp.device
                )
                info = torch.empty(batch_shape, dtype=torch.int32, device=inp.device)
                yield inp, {"out": (LU, pivots, info)}


@pytest.mark.linalg_lu_factor_ex_out
def test_linalg_lu_factor_ex_out():
    bench = LinalgLuFactorExOutBenchmark(
        op_name="linalg_lu_factor_ex_out",
        torch_op=_torch_lu_factor_ex,
        dtypes=_TEST_DTYPES,
    )
    if VENDOR == "ascend":
        bench.gems_op = flag_gems.linalg_lu_factor_ex_out
    bench.run()
