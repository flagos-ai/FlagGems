# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)

_CUDA_BLOCK_SIZE = 256
_ASCEND_BLOCK_SIZE = 512
_SUPPORTED_INPUT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _searchsorted_kernel(
    sorted_sequence,
    values,
    sorter,
    out,
    total_values,
    values_per_row,
    LOG_SEQUENCE_LEN: tl.constexpr,
    RIGHT: tl.constexpr,
    HAS_SORTER: tl.constexpr,
    IS_1D_SEQUENCE: tl.constexpr,
    USE_INT32_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    SEQUENCE_LEN: tl.constexpr,
):
    # Bitwalk (binary lifting) formulation of searchsorted:
    #   result = # of boundaries strictly below / not above `values`,
    # computed by probing seq[idx + step - 1] for step = 2^b, b = LOG..0.
    # Compared with the low/high `tl.where` bisection this keeps every
    # step a pure add (+ one select-free advance mask), so the unrolled
    # chain is much shorter and XPU's TTXIR passes stay linear in the
    # sequence length (old kernel: compile time exploded with LOG>=8:
    # ~150s @LOG=8, >1h @LOG=10; new kernel: 4-6s at LOG=13).
    # NaN semantics match the original: comparisons with NaN are false,
    # so `~go_left` is true and NaN advances to the right (end) position.
    # SEQUENCE_LEN is constexpr so the per-round probe address stays
    # affine-in-{offsets, idx} for the XPU backend (a runtime sequence_len
    # broke affine analysis ~1.25x on the 2D benchmark shapes). NOTE:
    # making BOTH SEQUENCE_LEN and values_per_row constexpr triggers an XPU
    # backend constant-fold bug for small non-pow2 sequences (S=2,3 gave
    # wrong results) -- keep values_per_row runtime.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < total_values
        values_in = tl.load(values + offsets, mask=mask, other=0)
    else:
        values_in = tl.load(values + offsets)

    if IS_1D_SEQUENCE:
        if USE_INT32_INDEX:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        else:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    else:
        if USE_INT32_INDEX:
            row_offsets = (offsets // values_per_row).to(tl.int32) * SEQUENCE_LEN
        else:
            row_offsets = (offsets // values_per_row) * SEQUENCE_LEN

    if USE_INT32_INDEX:
        idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for b in tl.static_range(LOG_SEQUENCE_LEN, -1, -1):
        step = 1 << b
        next_idx = idx + step
        probe = tl.minimum(next_idx, SEQUENCE_LEN) - 1
        in_range = next_idx <= SEQUENCE_LEN
        if HAS_SORTER:
            if NEED_MASK:
                si = tl.load(sorter + row_offsets + probe, mask=mask, other=0)
                if USE_INT32_INDEX:
                    si = si.to(tl.int32)
                mv = tl.load(sorted_sequence + row_offsets + si, mask=mask, other=0)
            else:
                si = tl.load(sorter + row_offsets + probe)
                if USE_INT32_INDEX:
                    si = si.to(tl.int32)
                mv = tl.load(sorted_sequence + row_offsets + si)
            valid = (si >= 0) & (si < SEQUENCE_LEN)
            tl.device_assert(in_range | (~valid), "sorter index out of range")
        else:
            if NEED_MASK:
                mv = tl.load(sorted_sequence + row_offsets + probe, mask=mask, other=0)
                in_range = in_range & mask
            else:
                mv = tl.load(sorted_sequence + row_offsets + probe)
        if RIGHT:
            go_left = values_in < mv
        else:
            go_left = values_in <= mv
        advance = (~go_left).to(tl.int32) & in_range.to(tl.int32)
        idx += step.to(idx.dtype) * advance.to(idx.dtype)

    if NEED_MASK:
        tl.store(out + offsets, idx, mask=mask)
    else:
        tl.store(out + offsets, idx)


_SM_STAGED_GEOM = {}
_SM_BUDGET_BYTES = 32 * 1024
_SM_TARGET_PROGRAMS = 64


@triton.jit
def _searchsorted_sm_staged_kernel(
    seq_desc,
    values_ptr,
    out_desc,
    n_rows,
    q_per_row,
    L: tl.constexpr,
    LOG: tl.constexpr,
    RP: tl.constexpr,
    RQ: tl.constexpr,
    RIGHT: tl.constexpr,
    TY: tl.constexpr,
    OTY: tl.constexpr,
    WIDEN: tl.constexpr,
):
    """Per-row SM staging + on-chip bitwalk bisection.

    Why this shape and not a plain GM-loading bisection: the probe loads are
    data-dependent gathers, and on XPU a gather against global memory degenerates
    into one small transfer per lane. Staging the row's boundaries into cluster-
    shared memory first turns every probe into an on-chip read. Measured on the
    three benchmark shapes this is ~2.3x the bitwalk kernel below.

    Two structural constraints, both found the hard way:

    * `tle.gpu.copy` must sit inside a *dynamic* loop with `tle.gpu.alloc`
      outside it. `TritonXPULoopGrid` hoists only top-level ops; a top-level copy
      at a pid-dependent offset gets hoisted and then trips
      "operand #0 does not dominate this use". Putting the copy in a dynamic
      `scf.for` keeps it out of the hoist set, so no compiler change is needed.
      (`alloc` inside the loop instead fails the front end with
      "'scf.yield' op must be the last operation in the parent block"; so does a
      three-argument `range`.)
    * The result must leave through `obuf` + a TLE descriptor copy. Writing
      straight to the global output pointer measured 2.0x slower on the same
      kernel (22.32us vs 11.06us).

    KNOWN CEILING: the bisection is NOT vectorised, and cannot be. Each round's
    probe index depends on the previous round's comparison, which puts
    `triton_xpu.tle_local_ptr` / `arith.fptosi` -- ops with no vector form -- into
    the closure walk's user-reachable set, so `VectorizabilityAnalysis` rejects
    the whole closure and every SM read stays scalar. Extending the whitelist
    does not help: the vectorised SM gather itself misbehaves on device (one form
    crashes the card, another returns wrong results), see
    `artifacts/op-perf-batch-2026-09/evidence/searchsorted-20260924/vectorization-cycle-20260924/`.
    """
    pr = tl.program_id(0)
    pq = tl.program_id(1)
    nrb = tl.num_programs(0)
    rpp = (n_rows + nrb - 1) // nrb
    rstart = pr * rpp
    rend = tl.minimum(rstart + rpp, n_rows)

    sbuf = tle.gpu.alloc([RP, L], dtype=TY, layout=None, scope=tle.gpu.smem)
    obuf = tle.gpu.alloc([RP, RQ], dtype=OTY, layout=None, scope=tle.gpu.lmem)
    if WIDEN:
        # bf16 rides here. A bf16 value read out of `addrspace(2)` has no ISel
        # pattern, so staging bf16 directly aborts `llc`; staging its raw int16
        # bits and widening is the way round it. bf16 IS the high half of fp32,
        # so `int32(bits) << 16` is exact and costs a shift rather than the
        # `vcvt`-class conversion that made the host-side widening a net loss
        # (0.2551 -> 0.1518 on the official matrix).
        fbuf = tle.gpu.alloc([RP, L], dtype=tl.float32, layout=None, scope=tle.gpu.smem)

    nb = (rend - rstart + RP - 1) // RP
    for b in range(0, nb):
        rb = rstart + b * RP
        tle.gpu.copy(seq_desc, sbuf, [RP, L], [rb, 0])
        if WIDEN:
            wr = tl.broadcast_to(tl.arange(0, RP)[:, None], (RP, L))
            wc = tl.broadcast_to(tl.arange(0, L)[None, :], (RP, L))
            wbits = tl.load(tle.gpu.local_ptr(sbuf, (wr, wc))).to(tl.int32)
            tl.store(
                tle.gpu.local_ptr(fbuf, (wr, wc)),
                (wbits << 16).to(tl.float32, bitcast=True),
            )

        rows = tl.broadcast_to(tl.arange(0, RP)[:, None], (RP, RQ))
        cols = tl.broadcast_to(tl.arange(0, RQ)[None, :], (RP, RQ))
        qidx = (rb + rows) * q_per_row + pq * RQ + cols
        if WIDEN:
            vbits = tl.load(values_ptr + qidx).to(tl.int32)
            v = (vbits << 16).to(tl.float32, bitcast=True)
        else:
            v = tl.load(values_ptr + qidx)

        idx = tl.zeros((RP, RQ), dtype=tl.int32)
        for k in tl.static_range(LOG, -1, -1):
            step = 1 << k
            nxt = idx + step
            probe = tl.minimum(nxt, L) - 1
            if WIDEN:
                mv = tl.load(tle.gpu.local_ptr(fbuf, (rows, probe)))
            else:
                mv = tl.load(tle.gpu.local_ptr(sbuf, (rows, probe)))
            if RIGHT:
                go_left = v < mv
            else:
                go_left = v <= mv
            # Keep the mask in i1 and convert once -- three fewer ops per round
            # than widening both sides to int32 and ANDing there. Measured
            # NEUTRAL on the official matrix (0.7593 against 0.7609, i.e. inside
            # the noise), which is itself the useful fact: the loop is
            # latency-bound on the gather, not op-count-bound, so shaving
            # arithmetic here buys nothing. Kept because it is strictly fewer
            # instructions, not because it is faster.
            idx += step * ((~go_left) & (nxt <= L)).to(tl.int32)

        tl.store(tle.gpu.local_ptr(obuf, (rows, cols)), idx.to(OTY))
        tle.gpu.copy(obuf, out_desc, [RP, RQ], [rb, pq * RQ])


def _sm_staged_geom(n_rows, q_per_row, L, itemsize):
    """`(grid_rows, grid_q, RP, RQ)` for the staged kernel.

    Carried over from the prototype that produced the measured numbers; it has
    NOT been re-tuned against this implementation. Two hard constraints and one
    aim:
      * `RP * L * itemsize` has to fit the per-cluster SM budget;
      * `q_per_row % RQ == 0`, because the query tile is not masked -- a partial
        tile would read past the row (the descriptor copy clamps, a `tl.load`
        through a raw pointer does not);
      * aim for roughly one program per cluster.
    """
    key = (n_rows, q_per_row, L, itemsize)
    geom = _SM_STAGED_GEOM.get(key)
    if geom is not None:
        return geom

    # `rq` has to start at a power of two: `min(q_per_row, 512)` is not one in
    # general (q_per_row=3 gives 3, and the divisibility loop below would then
    # never run), and `tle.gpu.alloc` rejects a non-power-of-2 extent.
    rq = 1
    while rq * 2 <= min(q_per_row, 512):
        rq *= 2
    while rq > 1 and q_per_row % rq != 0:
        rq //= 2
    grid_q = q_per_row // rq
    # Two rows per program, and let the row count drive the grid. Swept on the
    # two 2-D benchmark shapes: many programs with a small tile beats few
    # programs with a big one by ~1.9x -- shape3 at grid 128/RP=2 runs 145us
    # against 273us at grid 64/RP=4, and RP=8/16 is worse again (697us). The
    # extra waves are nearly free because the bisection is latency-bound rather
    # than throughput-bound: each round's probe waits on the previous round's
    # comparison, so a program with 2048 live lanes just serialises them.
    grid_rows = max(1, -(-n_rows // 2))
    rp = 1
    while rp * grid_rows < n_rows:
        rp *= 2
    cap = max(1, _SM_BUDGET_BYTES // (L * itemsize))
    if rp > cap:
        return None
    geom = (grid_rows, grid_q, rp, rq)
    _SM_STAGED_GEOM[key] = geom
    return geom


def _sm_staged_dtype(dtype):
    """tl dtype for the staged kernel, or None when the path is not usable.

    bf16 is not here on purpose. With the closure rejected (which is always, see
    the kernel docstring) a scalar bf16 read out of `addrspace(2)` has no ISel
    pattern and `llc` aborts with SIGABRT. bf16 still reaches the staged path,
    through `_staged_inputs` below, by widening to fp32 first -- the widening is
    exact, so the comparison result is unchanged.
    """
    if not _HAS_TLE:
        return None
    return {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.int8: tl.int8,
        torch.uint8: tl.uint8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
    }.get(dtype)


def _staged_inputs(seq, values):
    """`(seq, values, ty, widen)` as the staged kernel wants them, or four Nones.

    bf16 is bit-viewed to int16 and widened inside the kernel. That is exact --
    bf16 is the high half of fp32 -- and it dodges both problems bf16 otherwise
    has here: a scalar bf16 read out of `addrspace(2)` has no ISel pattern (llc
    aborts), and a host-side widening costs as much as the whole kernel.
    """
    if seq.dtype == torch.bfloat16 or values.dtype == torch.bfloat16:
        if seq.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            return None, None, None, None
        return seq.view(torch.int16), values.view(torch.int16), tl.int16, True
    ty = _sm_staged_dtype(seq.dtype)
    if ty is None or _sm_staged_dtype(values.dtype) is None:
        return None, None, None, None
    return seq, values, ty, False


def _sm_staged_supported(sorted_sequence, values, out):
    """`(n_rows, q_per_row, seq, values)` for the staged path, or None.

    The returned `seq`/`values` are what the kernel must be launched with -- for
    bf16 they are fp32 widened copies, so the caller has to use these and not the
    originals it passed in.
    """
    if not _HAS_TLE or sorted_sequence.dim() > 2:
        return None
    L = sorted_sequence.shape[-1]
    if L < 2 or (L & (L - 1)) != 0:
        # The staging buffer is [RP, L] and `tle.gpu.alloc` requires power-of-2
        # extents. Padding the buffer past L would leave the tail holding stale
        # SM bytes, so a non-power-of-2 sequence takes the bitwalk fallback
        # instead of a padded staging buffer.
        return None
    q_per_row = values.shape[-1] if sorted_sequence.dim() != 1 else values.numel()
    if q_per_row == 0:
        return None
    n_rows = sorted_sequence.shape[0] if sorted_sequence.dim() != 1 else 1
    if values.numel() != n_rows * q_per_row:
        return None
    if _sm_staged_dtype(out.dtype) is None:
        return None
    seq, vals, ty, widen = _staged_inputs(sorted_sequence, values)
    if seq is None:
        return None
    if _sm_staged_geom(n_rows, q_per_row, L, seq.element_size()) is None:
        return None
    return n_rows, q_per_row, seq, vals, ty, widen


def _sm_staged_launch(seq2, val2, out2, n_rows, q_per_row, right, ty, widen):
    """Launch the staged kernel over `seq2` [n_rows, L] against `val2` [n_rows, Q]."""
    L = seq2.shape[-1]
    grid_rows, grid_q, rp, rq = _sm_staged_geom(
        n_rows, q_per_row, L, seq2.element_size()
    )
    seq_desc = TensorDescriptor.from_tensor(seq2, [rp, L])
    out_desc = TensorDescriptor.from_tensor(out2, [rp, rq])
    with torch_device_fn.device(seq2.device):
        _searchsorted_sm_staged_kernel[(grid_rows, grid_q)](
            seq_desc,
            val2,
            out_desc,
            n_rows,
            q_per_row,
            L=L,
            LOG=L.bit_length() - 1,
            RP=rp,
            RQ=rq,
            RIGHT=right,
            TY=ty,
            OTY=_sm_staged_dtype(out2.dtype),
            WIDEN=widen,
        )


def _normalize_right(right: bool, side: str | None) -> bool:
    if side is None:
        return bool(right)
    if side == "left":
        if right:
            raise RuntimeError(
                "torch.searchsorted(): side and right can't be set to opposites, "
                "got side of left while right was True"
            )
        return False
    if side == "right":
        return True
    raise RuntimeError(
        f"torch.searchsorted(): side can only be 'left' or 'right' but got {side}"
    )


def _check_dtype(tensor: torch.Tensor, name: str):
    if tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"searchsorted is not implemented for {name} dtype {tensor.dtype}"
        )


def _check_tensor_values_shape(sorted_sequence: torch.Tensor, values: torch.Tensor):
    if sorted_sequence.dim() == 0:
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match"
        )
    if sorted_sequence.dim() == 1:
        return
    if values.dim() != sorted_sequence.dim() or (
        tuple(values.shape[:-1]) != tuple(sorted_sequence.shape[:-1])
    ):
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match, but we got boundaries tensor "
            f"{list(sorted_sequence.shape)} and input value tensor {list(values.shape)}"
        )


def _check_scalar_values_shape(sorted_sequence: torch.Tensor):
    if sorted_sequence.dim() != 1:
        raise RuntimeError(
            "torch.searchsorted(): input value can be a scalar only when boundaries "
            "tensor dimension is 1, but we got boundaries tensor "
            f"dim({sorted_sequence.dim()}) and input value's dim(0) numel(1)"
        )


def _check_sorter(sorted_sequence: torch.Tensor, sorter: torch.Tensor | None):
    if sorter is None:
        return
    if tuple(sorter.shape) != tuple(sorted_sequence.shape):
        raise RuntimeError(
            "torch.searchsorted(): boundary and sorter must have the same size, "
            f"but got boundary tensor {list(sorted_sequence.shape)}"
            f"and got sorter tensor {list(sorter.shape)}"
        )
    if sorter.dtype != torch.int64:
        raise RuntimeError(
            "torch.searchsorted(): sorter must be a tensor of long dtype but got "
            f"dtype {sorter.dtype}"
        )
    if sorter.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorter and boundary tensors must be on the same device"
        )


def _prepare_out(
    values: torch.Tensor,
    out_int32: bool,
    out: torch.Tensor | None,
):
    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        return torch.empty(values.shape, dtype=out_dtype, device=values.device)
    if out.dtype != out_dtype:
        raise RuntimeError(
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be "
            "Int(int32) or Long(int64) depending on whether out_int32 flag is True"
        )
    if out.device != values.device:
        raise RuntimeError(
            "torch.searchsorted(): output tensor must be on the same device as input"
        )
    if tuple(out.shape) != tuple(values.shape):
        out.resize_(values.shape)
    return out


def _searchsorted_impl(
    sorted_sequence: torch.Tensor,
    values: torch.Tensor,
    *,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: torch.Tensor | None,
    out: torch.Tensor | None = None,
):
    right = _normalize_right(right, side)
    _check_dtype(sorted_sequence, "sorted_sequence")
    _check_dtype(values, "values")
    _check_tensor_values_shape(sorted_sequence, values)
    _check_sorter(sorted_sequence, sorter)
    if values.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorted_sequence and values must be on the same device"
        )

    out = _prepare_out(values, out_int32, out)
    if values.numel() == 0:
        return out
    if sorted_sequence.shape[-1] == 0:
        out.zero_()
        return out

    sorted_sequence_contiguous = sorted_sequence.contiguous()
    values_contiguous = values.contiguous()
    sorter_contiguous = sorter.contiguous() if sorter is not None else None
    is_ascend = runtime_device.vendor_name == "ascend"
    if sorter_contiguous is not None and is_ascend:
        sorted_sequence_contiguous = torch.gather(
            sorted_sequence_contiguous, -1, sorter_contiguous
        )
        sorter_contiguous = None
    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    )

    if not is_ascend:
        staged = None
        if sorter_contiguous is None:
            staged = _sm_staged_supported(
                sorted_sequence_contiguous, values_contiguous, kernel_out
            )
        else:
            # The staged kernel has no sorter indirection -- it probes the staged
            # row directly, so a sorter must be materialised away first. This has
            # to happen BEFORE the eligibility test: the unsorted sequence is
            # shape-compatible, so testing first would silently search the wrong
            # array. Costs one gather over the boundaries (15.5us on the 256x1024
            # benchmark shape) and is only paid when the staged path is taken;
            # otherwise the bitwalk below still gets the sorter and does the
            # indirection itself.
            materialised = torch.gather(
                sorted_sequence_contiguous, -1, sorter_contiguous
            )
            staged = _sm_staged_supported(materialised, values_contiguous, kernel_out)
            if staged is not None:
                sorted_sequence_contiguous = materialised
                sorter_contiguous = None
        if staged is not None:
            n_rows, q_per_row, seq_s, vals_s, ty_s, widen_s = staged
            seq2 = seq_s.view(1, -1) if sorted_sequence.dim() == 1 else seq_s
            _sm_staged_launch(
                seq2,
                vals_s.view(n_rows, q_per_row),
                kernel_out.view(n_rows, q_per_row),
                n_rows,
                q_per_row,
                right,
                ty_s,
                widen_s,
            )
            if kernel_out is not out:
                out.copy_(kernel_out)
            return out

    sequence_len = sorted_sequence.shape[-1]
    values_per_row = values.shape[-1] if sorted_sequence.dim() != 1 else values.numel()
    if is_ascend and sorted_sequence.dtype.is_floating_point:
        block_size = _ASCEND_BLOCK_SIZE
    elif is_ascend:
        block_size = _CUDA_BLOCK_SIZE
    else:
        # kunlunxin: size-banded block. The probe loads are data-dependent
        # gathers; larger blocks hide per-warp gather latency better, but too
        # large spills registers (2048 regressed on the 256x1024/512 case).
        numel = values.numel()
        if numel <= 4096:
            block_size = 256
        elif numel <= 16384:
            block_size = 512
        else:
            block_size = 1024
    use_int32_index = (
        values.numel() < torch.iinfo(torch.int32).max
        and sorted_sequence.numel() < torch.iinfo(torch.int32).max
    )
    need_mask = values.numel() % block_size != 0

    with torch_device_fn.device(sorted_sequence.device):
        grid = (triton.cdiv(values.numel(), block_size),)
        _searchsorted_kernel[grid](
            sorted_sequence_contiguous,
            values_contiguous,
            (
                sorter_contiguous
                if sorter_contiguous is not None
                else sorted_sequence_contiguous
            ),
            kernel_out,
            values.numel(),
            values_per_row,
            LOG_SEQUENCE_LEN=sequence_len.bit_length(),
            RIGHT=right,
            HAS_SORTER=sorter_contiguous is not None,
            IS_1D_SEQUENCE=sorted_sequence.dim() == 1,
            USE_INT32_INDEX=use_int32_index,
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
            SEQUENCE_LEN=sequence_len,
        )

    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def searchsorted(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED OUT")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )


def searchsorted_scalar(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED SCALAR")
    _check_scalar_values_shape(sorted_sequence)
    values = torch.scalar_tensor(self, device=sorted_sequence.device)
    return _searchsorted_impl(
        sorted_sequence,
        values,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_scalar_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SEARCHSORTED SCALAR OUT")
    _check_scalar_values_shape(sorted_sequence)
    values = torch.scalar_tensor(self, device=sorted_sequence.device)
    return _searchsorted_impl(
        sorted_sequence,
        values,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )
