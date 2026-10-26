import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.ops.log_softmax import log_softmax as common_log_softmax
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim

from ..heuristics_config_utils import HAS_SMT_RVISA, _sr_call, f16, f32, smt_rvisa_enabled
from ..heuristics_config_utils import tle as srv

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp
log = tl_extra_shim.log


@libentry()
@triton.heuristics(
    {
        # log_softmax_spacemit's tune config was static (ROW_SIZE=1,
        # COL_SIZE=256); both are made adaptive: ROW_SIZE keeps the grid within
        # the 512 programs this stack's runtime executes normally, and
        # COL_SIZE covers a whole row so the kernel takes the single-pass
        # branch -- the two-pass online path carries its max/denominator
        # reductions across loop iterations, which lowers to per-iteration DDR
        # scratch traffic (~100x wall time at n_cols=2048)
        "ROW_SIZE": lambda args: max(1, triton.cdiv(args["n_rows"], 512)),
        "COL_SIZE": lambda args: min(triton.next_power_of_2(args["n_cols"]), 8192),
        "ONE_TILE_PER_ROW": lambda args: min(triton.next_power_of_2(args["n_cols"]), 8192)
        >= args["n_cols"],
    }
)
@triton.jit
def log_softmax_kernel_spacemit(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
    ONE_TILE_PER_ROW: tl.constexpr,
):
    element_ty = output_ptr.type.element_ty
    row_start = tl.program_id(0) * ROW_SIZE
    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            if ONE_TILE_PER_ROW:
                # Fast path for short rows: keep the reduction 1-D, but batch
                # ROW_SIZE rows per program to reduce program dispatch overhead.
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(0,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(0,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                row_max = tl.max(row, axis=0)
                numerator = exp(row - row_max)
                denominator = tl.sum(numerator, axis=0)
                # a scalar (0-d) log lowers to the slow unstructured path on
                # this stack; broadcast to a 32-lane vector and average the
                # identical lanes (exact: d*32 and /32 are power-of-2 ops)
                log_denom = tl.sum(log(denominator + tl.zeros([32], tl.float32)), axis=0) / 32
                result = (row - row_max - log_denom).to(element_ty)
                tl.store(output_block_ptr, result, boundary_check=(0,))
            else:
                # Two-pass online log_softmax for large n_cols.
                row_max_total = tl.full((1,), value=-float("inf"), dtype=tl.float32)
                slow_denominator = tl.zeros((1,), dtype=tl.float32)

                for col_idx in range(0, n_cols, COL_SIZE):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr + row_idx * input_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    row = tl.load(
                        input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                    ).to(tl.float32)
                    block_max = tl.max(row, axis=0)
                    new_max = tl.maximum(row_max_total, block_max)
                    slow_denominator = slow_denominator * exp(row_max_total - new_max)
                    slow_denominator += tl.sum(exp(row - new_max), axis=0)
                    row_max_total = new_max

                slow_log_denom = log(slow_denominator)

                for col_idx in range(0, n_cols, COL_SIZE):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr + row_idx * input_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    output_block_ptr = tl.make_block_ptr(
                        base=output_ptr + row_idx * output_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    row = tl.load(
                        input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                    ).to(tl.float32)
                    result = (row - row_max_total - slow_log_denom).to(element_ty)
                    tl.store(output_block_ptr, result, boundary_check=(0,))


def _spacemit_log_softmax_lastdim(inp, out):
    n_rows, n_cols = inp.shape
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROW_SIZE"]),)
    with torch_device_fn.device(inp.device):
        log_softmax_kernel_spacemit[grid](
            out,
            inp,
            inp.stride(0),
            out.stride(0),
            n_rows,
            n_cols,
        )


if HAS_SMT_RVISA:

    # three-pass in-register log_softmax: (x - max) - log(sum(exp(x - max))),
    # ported from spine-triton python/tests/raw/test_raw_log_softmax.py with
    # the mv svector row-block host; 2.2-7.2x over log_softmax_kernel_spacemit,
    # whose two-pass online path collapses at n_cols > 8192 (K3 raw_vs_ops
    # bench, 2026-10-23). Variants cover the (input, output) dtype pairs
    # log_softmax() can produce: f16->f16, f16->f32 (half_to_float) and
    # f32->f32; math is always f32.
    @srv.raw_kernel
    def _log_softmax_rvisa_f16o16(
        X: srv.mem(f16),
        out: srv.mem(f16, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            acc_max = srv.cast(srv.vload(X, base), f32)
            for i in srv.range(0, Nfloor, nvl):
                va = srv.cast(srv.vload(X, base + i), f32)
                acc_max = srv.vmax(acc_max, va)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                # -65504 = lowest finite f16; exp of it underflows to 0, so
                # padded lanes never inflate the denominator
                ta = srv.cast(srv.vload(X, base + i, fill=-65504.0), f32)
                acc_max = srv.vmax(acc_max, ta)
            xmax = srv.vreduce_max(acc_max)
            acc_sum = srv.vzero(f32)
            for i in srv.range(0, Nfloor, nvl):
                vb = srv.cast(srv.vload(X, base + i), f32)
                acc_sum = acc_sum + srv.vexp(vb - xmax)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tb = srv.cast(srv.vload(X, base + i, fill=-65504.0), f32)
                acc_sum = acc_sum + srv.vexp(tb - xmax)
            denom = srv.vreduce_sum(acc_sum)
            log_denom = srv.vlog(denom)
            for i in srv.range(0, Nfloor, nvl):
                vc = srv.cast(srv.vload(X, base + i), f32)
                srv.vstore(out, base + i, srv.cast((vc - xmax) - log_denom, f16))
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tc = srv.cast(srv.vload(X, base + i), f32)
                srv.vstore(out, base + i, srv.cast((tc - xmax) - log_denom, f16))

    @srv.raw_kernel
    def _log_softmax_rvisa_f16o32(
        X: srv.mem(f16),
        out: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            acc_max = srv.cast(srv.vload(X, base), f32)
            for i in srv.range(0, Nfloor, nvl):
                va = srv.cast(srv.vload(X, base + i), f32)
                acc_max = srv.vmax(acc_max, va)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                ta = srv.cast(srv.vload(X, base + i, fill=-65504.0), f32)
                acc_max = srv.vmax(acc_max, ta)
            xmax = srv.vreduce_max(acc_max)
            acc_sum = srv.vzero(f32)
            for i in srv.range(0, Nfloor, nvl):
                vb = srv.cast(srv.vload(X, base + i), f32)
                acc_sum = acc_sum + srv.vexp(vb - xmax)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tb = srv.cast(srv.vload(X, base + i, fill=-65504.0), f32)
                acc_sum = acc_sum + srv.vexp(tb - xmax)
            denom = srv.vreduce_sum(acc_sum)
            log_denom = srv.vlog(denom)
            for i in srv.range(0, Nfloor, nvl):
                vc = srv.cast(srv.vload(X, base + i), f32)
                srv.vstore(out, base + i, (vc - xmax) - log_denom)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tc = srv.cast(srv.vload(X, base + i), f32)
                srv.vstore(out, base + i, (tc - xmax) - log_denom)

    @srv.raw_kernel
    def _log_softmax_rvisa_f32o32(
        X: srv.mem(f32),
        out: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            acc_max = srv.vload(X, base, dtype=f32)
            for i in srv.range(0, Nfloor, nvl):
                va = srv.vload(X, base + i, dtype=f32)
                acc_max = srv.vmax(acc_max, va)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                ta = srv.vload(X, base + i, dtype=f32, fill=-1e38)
                acc_max = srv.vmax(acc_max, ta)
            xmax = srv.vreduce_max(acc_max)
            acc_sum = srv.vzero(f32)
            for i in srv.range(0, Nfloor, nvl):
                vb = srv.vload(X, base + i, dtype=f32)
                acc_sum = acc_sum + srv.vexp(vb - xmax)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                # fill=-1e38: padded lanes get exp(-1e38 - xmax) ~= 0
                tb = srv.vload(X, base + i, dtype=f32, fill=-1e38)
                acc_sum = acc_sum + srv.vexp(tb - xmax)
            denom = srv.vreduce_sum(acc_sum)
            log_denom = srv.vlog(denom)
            for i in srv.range(0, Nfloor, nvl):
                vc = srv.vload(X, base + i, dtype=f32)
                srv.vstore(out, base + i, (vc - xmax) - log_denom)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tc = srv.vload(X, base + i, dtype=f32)
                srv.vstore(out, base + i, (tc - xmax) - log_denom)

    @triton.jit(do_not_specialize=["n_rows", "n_cols"])
    def _log_softmax_rvisa_host(X, out, n_rows, n_cols, BLOCK: tl.constexpr, VARIANT: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        row_end = min(row_base + BLOCK, n_rows)
        if VARIANT == 0:
            _sr_call(_log_softmax_rvisa_f16o16, outputs=[], inputs=[X, out, n_cols, row_base, row_end])
        elif VARIANT == 1:
            _sr_call(_log_softmax_rvisa_f16o32, outputs=[], inputs=[X, out, n_cols, row_base, row_end])
        else:
            _sr_call(_log_softmax_rvisa_f32o32, outputs=[], inputs=[X, out, n_cols, row_base, row_end])

    def _log_softmax_rvisa_lastdim(inp, out):
        n_rows, n_cols = inp.shape
        # BLOCK rows per program keeps the grid within the 512 programs the
        # runtime executes
        block = max(1, triton.cdiv(n_rows, 512))
        if inp.dtype == torch.float16:
            variant = 0 if out.dtype == torch.float16 else 1
        else:
            variant = 2
        with torch_device_fn.device(inp.device):
            _log_softmax_rvisa_host[(triton.cdiv(n_rows, block),)](
                inp, out, n_rows, n_cols, BLOCK=block, VARIANT=variant
            )


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_SPACEMIT LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    if dim != self.ndim - 1:
        return common_log_softmax(self, dim, half_to_float)

    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    inp = self.contiguous()

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)
    # the streaming kernel seeds its max with one full first tile, so it needs
    # n_cols >= 64 (the widest VLMAX) to not read into the next row
    if (
        smt_rvisa_enabled()
        and self.dtype in (torch.float16, torch.float32)
        and n_rows > 0
        and n_cols >= 64
    ):
        out_2d = torch.empty_like(inp_2d, dtype=dtype)
        _log_softmax_rvisa_lastdim(inp_2d, out_2d)
        return out_2d.view_as(inp)
    out_2d = torch.empty_like(inp_2d, dtype=dtype)
    _spacemit_log_softmax_lastdim(inp_2d, out_2d)
    return out_2d.view_as(inp)
