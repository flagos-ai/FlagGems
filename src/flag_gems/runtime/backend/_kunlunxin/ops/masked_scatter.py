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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# masked_scatter is  out[p] = where(mask[p], source[cumsum(mask)[p]-1], inp[p]).
#
# N > 4096 fast path (this file): a VENDOR-STYLE 2-LAUNCH, 768-thread, contiguous
# PARTITION model (ported from masked_scatter.xpu stage_1/stage_2):
#   stage_1: each of GRID*64 = 768 threads owns a 64-aligned contiguous partition
#            of N, counts its Trues, then an intra-cluster __shared__ prefix writes
#            per-thread counts tcnt[768] and per-cluster totals ctot[12] -- NO big
#            cumsum, NO base[] GM round-trip.
#   stage_2: each thread derives its global source base = sum(ctot[0..cluster-1])
#            + intra-cluster prefix(tcnt), then scatters its partition with the
#            rolling-source compaction loop.
# This keeps all 768 threads busy even for tiny N (fixes the 256^2/1024^2 launch-
# overhead floor) and removes a whole launch + two GM arrays vs the old 3-launch
# (count -> counts[] -> scan -> base[] -> where).  GRID=12 == the 12 physical
# clusters of this XPU (sibling histc tle.raw runs GRID=12 with exact sums, which
# requires all 768 workers; torch multi_processor_count==8 under-reports).


@libentry()
@triton.jit
def masked_scatter_single_block_kernel(
    out_ptr, inp_ptr, mask_ptr, src_ptr, N, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    inp_val = tl.load(inp_ptr + offsets, mask=block_mask, other=0)
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    src_indices = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    active = block_mask & mask_val
    src_val = tl.load(src_ptr + src_indices, mask=active, other=0)
    out_val = tl.where(mask_val, src_val, inp_val)
    tl.store(out_ptr + offsets, out_val, mask=block_mask)


_MS_GRID = 12  # 12 physical clusters; grid=16/24 would drop clusters -> never exceed
_MS_TLE = {"init": False, "ok": False, "stage1": None, "stage2": {}}


# ---------------------------------------------------------------------------
# tle.raw payload sources live in ../payload/src/*.xpu and ship as the
# secrecy-hardened device object ../payload/obj/masked_scatter.o (packed with
# docs/xpu3/how_to_pack_payload/pack_payload.py --obj-dir, then merged from the
# per-entry .o with `ld.lld -r`).  The C++ is deliberately NOT kept in this
# file.  Entry symbols (stub name == entry symbol, signature == C++ ABI):
#   ms_stage1        <- payload/src/ms_stage1.xpu
#   ms_stage2_int    <- payload/src/ms_stage2_int.xpu    (elt=int,   WT=1024, KT=256)
#   ms_stage2_short  <- payload/src/ms_stage2_short.xpu  (elt=short, WT=1024, KT=512)
# ---------------------------------------------------------------------------
def _ms_tle_init():
    if _MS_TLE["init"]:
        return _MS_TLE["ok"]
    _MS_TLE["init"] = True
    try:
        import os as _os

        import triton as _triton
        import triton.experimental.tle as _tle

        # We ship the .o (not the .xpu source): the stub name must equal the
        # entry symbol and the signature must match the C++ ABI.  In object=
        # mode the merge-time signature check is skipped (the payload isn't in
        # IR), so a wrong signature fails at runtime, not compile -- keep the
        # stub in lockstep with the C++.  Re-pack after editing any .xpu: the
        # compile-cache key is the .o content digest, so a stale .o silently
        # keeps the old code.
        _PAY_OBJ = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "payload",
            "obj",
        )
        # Both stages live in one merged object, so one path serves every stub.
        _MS_OBJ = _os.path.join(_PAY_OBJ, "masked_scatter.o")

        s1ns = {"_tle": _tle, "_OBJ": _MS_OBJ}
        exec(
            "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
            "def ms_stage1(tcnt, ctot, mask, N):\n"
            "    ...\n",
            s1ns,
        )
        _s1callee = s1ns["ms_stage1"]

        @_triton.jit
        def _s1(T, C, M, N):
            _tle.raw.call(_s1callee, (T, C, M, N))

        _MS_TLE["stage1"] = _s1

        for elt in ("int", "short"):
            sym = f"ms_stage2_{elt}"
            ns = {"_tle": _tle, "_OBJ": _MS_OBJ}
            exec(
                "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
                f"def {sym}(out, inp, mask, src, tcnt, ctot, N, ELT):\n"
                "    ...\n",
                ns,
            )
            callee = ns[sym]

            def _mk(callee=callee):
                @_triton.jit
                def _k(OUT, INP, M, S, T, C, N, ELT):
                    _tle.raw.call(callee, (OUT, INP, M, S, T, C, N, ELT))

                return _k

            _MS_TLE["stage2"][elt] = _mk()
        _MS_TLE["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER tle.raw init failed: %r", _e)
        _MS_TLE["ok"] = False
    return _MS_TLE["ok"]


@libentry()
@triton.jit
def mask_block_count_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    sel = tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(sel, axis=0))


@libentry()
@triton.jit
def masked_scatter_where_kernel(
    out_ptr, inp_ptr, mask_ptr, src_ptr, base_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    base = tl.load(base_ptr + pid)
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    src_indices = base + tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    active = block_mask & mask_val
    inp_val = tl.load(inp_ptr + offsets, mask=block_mask, other=0)
    src_val = tl.load(src_ptr + src_indices, mask=active, other=0)
    tl.store(out_ptr + offsets, tl.where(mask_val, src_val, inp_val), mask=block_mask)


def _tle_ms_where(out, inp, mask, src, N):
    """2-launch partition masked_scatter; None -> defer to the codegen kernels."""
    try:
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None
        if N >= 2**31:
            return None
        if not (
            out.is_contiguous()
            and inp.is_contiguous()
            and mask.is_contiguous()
            and src.is_contiguous()
        ):
            return None
        if not _ms_tle_init():
            return None
        elt = "int" if inp.dtype == torch.float32 else "short"
        s2 = _MS_TLE["stage2"].get(elt)
        if s2 is None:
            return None
        u8mask = mask.view(torch.uint8)
        nthreads = _MS_GRID * 64
        tcnt = torch.empty(nthreads, dtype=torch.int32, device=inp.device)
        ctot = torch.empty(_MS_GRID * 16, dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _MS_TLE["stage1"][(_MS_GRID,)](tcnt, ctot, u8mask, N)
            if elt == "int":
                OUT, INP, S = out, inp, src
            else:
                OUT = out.view(torch.int16)
                INP = inp.view(torch.int16)
                S = src.view(torch.int16)
            s2[(_MS_GRID,)](OUT, INP, u8mask, S, tcnt, ctot, N, inp.element_size())
        return out
    except Exception as _e:  # pragma: no cover - never break correctness
        logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER tle path deferred: %r", _e)
        return None


def masked_scatter_impl(out, inp, mask, source, N):
    flat_out = out.ravel()
    flat_inp = inp.ravel()
    flat_mask = mask.ravel()
    flat_src = source.ravel()
    # an all-False mask yields an empty `source`; guard so a zero-size buffer
    # never reaches the kernels (values are discarded by where for False).
    if flat_src.numel() == 0:
        flat_src = torch.zeros(1, dtype=source.dtype, device=source.device)

    if N == 0:
        return out

    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        with torch_device_fn.device(inp.device):
            masked_scatter_single_block_kernel[(1,)](
                flat_out,
                flat_inp,
                flat_mask,
                flat_src,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return out

    # 2-launch partition fast path.  Returns None -> defer to the codegen kernels.
    res = _tle_ms_where(flat_out, flat_inp, flat_mask, flat_src, N)
    if res is not None:
        return out

    BLOCK_SIZE = 8192
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    idx_dtype = torch.int32 if N < 2**31 else torch.int64
    with torch_device_fn.device(inp.device):
        counts = torch.empty(n_blocks, dtype=idx_dtype, device=inp.device)
        mask_block_count_kernel[(n_blocks,)](
            flat_mask, counts, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=16
        )
        base = torch.zeros(n_blocks, dtype=idx_dtype, device=inp.device)
        torch.cumsum(counts[:-1], dim=0, out=base[1:])
        masked_scatter_where_kernel[(n_blocks,)](
            flat_out,
            flat_inp,
            flat_mask,
            flat_src,
            base,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,
        )
    return out


def masked_scatter(inp, mask, source):
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER")

    assert broadcastable(
        inp.shape, mask.shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"

    _, mask = torch.broadcast_tensors(inp, mask)

    out = torch.empty_like(inp)
    if not out.is_contiguous():
        out = out.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if not source.is_contiguous():
        source = source.contiguous()
    if not inp.is_contiguous():
        inp = inp.contiguous()

    N = out.numel()
    masked_scatter_impl(out, inp, mask, source, N)
    return out


def masked_scatter_(inp, mask, source):
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER_")

    assert broadcastable(inp.shape, mask.shape)
    _, mask = torch.broadcast_tensors(inp, mask)

    if not inp.is_contiguous():
        raise RuntimeError(
            "in-place operation currently requires contiguous input tensor. "
        )

    mask = mask if mask.is_contiguous() else mask.contiguous()
    source = source if source.is_contiguous() else source.contiguous()

    N = inp.numel()
    masked_scatter_impl(inp, inp, mask, source, N)
    return inp
