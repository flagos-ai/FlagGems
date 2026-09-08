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

LINALG_LU_FACTOR_SHAPE = [
    [16, 16],
    [32, 32],
    [64, 64],
    [128, 128],
    [256, 256],
    [1024, 512],
    [32, 16],
    [16, 32],
    [128, 64],
    [64, 128],
    [4, 32, 32],
    [128, 16, 16],
    [1024, 512, 512],
]

LinalgLUFactorResult = namedtuple("LinalgLUFactorResult", ["LU", "pivots"])


def _swap_rows(lu, i, pivot_row):
    *batch_shape, m, n = lu.shape
    device = lu.device

    rows = torch.arange(m, device=device).expand(*batch_shape, -1)

    mask_i = (rows == i).float().unsqueeze(-1)
    mask_p = (rows == pivot_row.unsqueeze(-1)).float().unsqueeze(-1)

    row_i_vals = (lu * mask_i).sum(dim=-2, keepdim=True)
    row_p_vals = (lu * mask_p).sum(dim=-2, keepdim=True)

    mask_i_full = mask_i.expand(*batch_shape, m, n)
    mask_p_full = mask_p.expand(*batch_shape, m, n)
    diff_ip = (row_p_vals - row_i_vals).expand(*batch_shape, m, n)
    diff_pi = (row_i_vals - row_p_vals).expand(*batch_shape, m, n)

    lu = lu + mask_i_full * diff_ip
    lu = lu + mask_p_full * diff_pi
    return lu


def _lu_factor_pivot(lu, m, n, k):
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


def ops_lu_factor(input, *, pivot=True):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions"
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

    return LinalgLUFactorResult(lu, pivots)


if VENDOR == "ascend":

    def _torch_lu_factor(inp, *, pivot=True, out=None):
        """Vendor-aware wrapper: uses torch ops on Ascend, native on other vendors."""
        res = ops_lu_factor(inp, pivot=pivot)
        if out is not None:
            lu_out, piv_out = out
            lu_out.copy_(res.LU)
            piv_out.copy_(res.pivots)
            return lu_out, piv_out
        return res.LU, res.pivots

else:
    _torch_lu_factor = torch.linalg.lu_factor


class LinalgLuFactorBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = _TEST_DTYPES
    # mode = base.Config.mode if VENDOR != "ascend" else consts.BenchMode.OPERATOR

    def get_input_iter(self, dtype):
        for inp_shape in LINALG_LU_FACTOR_SHAPE:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                yield inp, {"pivot": pivot}


@pytest.mark.linalg_lu_factor
def test_linalg_lu_factor():
    bench = LinalgLuFactorBenchmark(
        op_name="linalg_lu_factor",
        torch_op=_torch_lu_factor,
        gems_op=flag_gems.linalg_lu_factor,
        dtypes=_TEST_DTYPES,
    )
    bench.run()


class LinalgLuFactorOutBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = _TEST_DTYPES

    def get_input_iter(self, dtype):
        for inp_shape in LINALG_LU_FACTOR_SHAPE:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                k = min(inp_shape[-2], inp_shape[-1])
                batch_shape = inp_shape[:-2]
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                LU = torch.empty(inp.shape, dtype=dtype, device=inp.device)
                pivots = torch.empty(
                    (*batch_shape, k), dtype=torch.int32, device=inp.device
                )
                yield inp, {"pivot": pivot}, {"out": (LU, pivots)}


@pytest.mark.linalg_lu_factor_out
def test_linalg_lu_factor_out():
    bench = LinalgLuFactorOutBenchmark(
        op_name="linalg_lu_factor_out",
        torch_op=_torch_lu_factor,
        dtypes=_TEST_DTYPES,
    )
    if VENDOR == "ascend":
        bench.gems_op = flag_gems.linalg_lu_factor_out
    bench.run()
