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

if VENDOR == "ascend":
    Config.mode = consts.BenchMode.OPERATOR

# Use the same shapes as linalg_lu_factor for consistency
LU_SHAPES = [
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

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])


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


def _lu_factor_pivot(lu, m, n, k):
    """LU factorization with partial pivoting."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)

    for i in range(k):
        col = lu[..., i:, i].abs()
        pivot_rel = torch.argmax(col, dim=-1)
        pivot_row = pivot_rel + i
        pivots[..., i] = (pivot_row + 1).to(torch.int32)

        lu = _swap_rows(lu, i, pivot_row)

        pivot_val = lu[..., i, i]
        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    return lu, pivots


def _lu_factor_no_pivot(lu, m, n, k):
    """LU factorization without pivoting."""
    *batch_shape, _, _ = lu.shape
    device = lu.device
    pivots = torch.empty((*batch_shape, k), dtype=torch.int32, device=device)

    for i in range(k):
        pivots[..., i] = i + 1
        pivot_val = lu[..., i, i]
        lu[..., i + 1 :, i] = lu[..., i + 1 :, i] / pivot_val.unsqueeze(-1)

        if i + 1 < m and i + 1 < n:
            l_col = lu[..., i + 1 :, i].unsqueeze(-1)
            u_row = lu[..., i : i + 1, i + 1 :]
            lu[..., i + 1 :, i + 1 :] = lu[..., i + 1 :, i + 1 :] - l_col @ u_row

    return lu, pivots


def _unpack_lu_manual(lu, pivots, m, n, k, pivot):
    """Manual unpack of (lu, pivots) into (P, L, U).

    P is built by applying the row swaps to the identity in REVERSE order
    (i = k-1 .. 0), matching torch.linalg.lu semantics (see
    tests/test_linalg_lu.py for details).
    """
    device = lu.device
    dtype = lu.dtype
    batch_shape = lu.shape[:-2]

    ll = lu[..., :, :k].tril()
    diag = torch.arange(k, device=device)
    ll[..., diag, diag] = 1
    u = lu[..., :k, :].triu()

    if not pivot:
        return torch.empty(0, device=device, dtype=dtype), ll, u

    P = torch.zeros((*batch_shape, m, m), device=device, dtype=dtype)
    diag_m = torch.arange(m, device=device)
    P[..., diag_m, diag_m] = 1
    for i in range(k - 1, -1, -1):
        P = _swap_rows(P, i, (pivots[..., i] - 1).to(torch.int64))
    return P, ll, u


def ops_lu(input, *, pivot=True):
    """Manual Python reference for linalg_lu."""
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu: Expected input to have at least 2 dimensions"
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
        lu, pivots = _lu_factor_pivot(lu, m, n, k)
    else:
        lu, pivots = _lu_factor_no_pivot(lu, m, n, k)

    P, L, U = _unpack_lu_manual(lu, pivots, m, n, k, pivot)
    return LinalgLUResult(P, L, U)


if VENDOR == "ascend":

    def _torch_lu(inp, *, pivot=True, out=None):
        """Vendor-aware wrapper: uses manual ops on Ascend, native on other vendors.

        Runs the Python reference directly on the NPU input, the same way the
        linalg_lu_factor benchmark reference does.
        """
        res = ops_lu(inp, pivot=pivot)
        if out is not None:
            p_out, l_out, u_out = out
            p_out.resize_(res.P.shape)
            p_out.copy_(res.P)
            l_out.resize_(res.L.shape)
            l_out.copy_(res.L)
            u_out.resize_(res.U.shape)
            u_out.copy_(res.U)
            return p_out, l_out, u_out
        return res.P, res.L, res.U

else:
    _torch_lu = torch.linalg.lu


class LinalgLuBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_SHAPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                yield inp, {"pivot": pivot}


@pytest.mark.linalg_lu
def test_linalg_lu():
    bench = LinalgLuBenchmark(
        op_name="linalg_lu",
        torch_op=_torch_lu,
        gems_op=flag_gems.linalg_lu,
        dtypes=_TEST_DTYPES,
    )
    bench.run()


class LinalgLuOutBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_SHAPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                if pivot and VENDOR != "ascend":
                    # out variant for pivot=True already covered by main benchmark
                    continue
                m, n = inp_shape[-2], inp_shape[-1]
                k = min(m, n)
                batch_shape = inp_shape[:-2]
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                P = torch.empty((*batch_shape, m, m), dtype=dtype, device=inp.device)
                L = torch.empty((*batch_shape, m, k), dtype=dtype, device=inp.device)
                U = torch.empty((*batch_shape, k, n), dtype=dtype, device=inp.device)
                yield inp, {"out": (P, L, U)}


@pytest.mark.linalg_lu_out
def test_linalg_lu_out():
    bench = LinalgLuOutBenchmark(
        op_name="linalg_lu_out",
        torch_op=_torch_lu,
        dtypes=_TEST_DTYPES,
    )
    if VENDOR == "ascend":
        bench.gems_op = flag_gems.linalg_lu_out
    bench.run()
