# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared INT8 matmul with a guarded, on-chip Fixpipe epilogue.

Packing produces a per-column Fixpipe scale and sparse diagonal row scales.
Both scales are applied in the kernel. A device flag selects an FP32-compatible
scalar fallback for values outside the FP16 intermediate's safe range.
"""

import fcntl
import hashlib
import importlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems.utils import triton_lang_extension as ext

from . import compile_fixpipe as helper

ROOT = Path(__file__).parent
CACHE = Path(tempfile.gettempdir()) / "flaggems_mm_aic"
READY = set()
_REGISTER_LOCK = threading.RLock()


def _register_impl(m, n, k, mp, np, bm, bn, bk, cores, input_nz, batch_rows, prefetch):
    local_bytes = (bm + bn) * k + n * 8 + bm * bn * 2 + (bm // 16 + 1) * 512
    assert local_bytes <= 512 * 1024
    assert bm * bm * 2 <= 65536 and bm * bn * 2 <= 65536
    assert bm * bk <= 65536 and bn * bk <= 65536
    extra = 16 if bm % 32 == 0 and local_bytes + 16 * k <= 512 * 1024 else 0
    source = (ROOT / "mm_aic.cpp").read_bytes()
    header = (ROOT / "mm_aic_soft_float.hpp").read_bytes()
    compiler = Path(helper._find_ccec())
    identity = repr(
        (
            m,
            n,
            k,
            mp,
            np,
            bm,
            bn,
            bk,
            cores,
            input_nz,
            batch_rows,
            prefetch,
            str(compiler),
            compiler.stat().st_mtime_ns,
        )
    ).encode()
    revision = hashlib.sha256(
        source
        + header
        + identity
        + Path(__file__).read_bytes()
        + Path(helper.__file__).read_bytes()
    ).hexdigest()
    name = "mm_aic_" + revision[:24]
    if name in READY:
        return name, revision
    CACHE.mkdir(parents=True, exist_ok=True)
    cpp = CACHE / (name + ".cpp")
    bc = CACHE / (name + ".bc")
    defines = dict(
        M=m,
        N=n,
        K=k,
        MP=mp,
        NP=np,
        BM=bm,
        AM=bm + extra,
        BN=bn,
        BK=bk,
        CORES=cores,
        INPUT_NZ=int(input_nz),
        BATCH_ROWS=int(batch_rows),
        PREFETCH_INPUT=int(prefetch),
        CUBE_ENTRY="_mlir_ciface_" + name,
    )
    source_bytes = (
        "".join(
            f"#define CV_{key} {value}\n" for key, value in defines.items()
        ).encode()
        + source
    )
    with (CACHE / (name + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cpp.exists() or cpp.read_bytes() != source_bytes:
            cpp.write_bytes(source_bytes)
        if not bc.exists() or bc.stat().st_size == 0:
            temporary = bc.with_name(bc.name + f".{os.getpid()}.tmp")
            try:
                subprocess.check_call(
                    helper.compile_cmd(cpp, temporary) + ["-O3", "-I", str(ROOT)]
                )
                os.replace(temporary, bc)
            finally:
                temporary.unlink(missing_ok=True)

    def init(self, a, b, o, deq, diag, flag, sa, sb, ws, pid, out=None):
        assert out is not None
        self.arg_type["pid"] = tl.int32

    op = type(
        name,
        (),
        dict(
            name=name,
            core=al.CORE.CUBE,
            pipe=al.PIPE.PIPE_ALL,
            mode=al.MODE.SIMD,
            symbol=name,
            bitcode=str(bc),
            source=str(cpp),
            extra_attr="flaggems_cube_only=true",
            compile=helper.makefile_compile().replace(
                "-std=c++17", f"-std=c++17 -O3 -I{ROOT}"
            ),
            __init__=init,
        ),
    )
    al.register_custom_op(op)
    READY.add(name)
    return name, revision


def _register(*args):
    with _REGISTER_LOCK:
        return _register_impl(*args)


@triton.jit
def _mm_aic_kernel(
    A, B, OUT, DQ, DG, FLAG, SA, SB, WS, OP: tl.constexpr, REV: tl.constexpr
):
    with al.scope(core_mode="cube"):
        # The external fragment writes GM directly; its formal tensor output
        # only describes the custom-op ABI and is never read.
        dummy = tl.load(A + tl.arange(0, 16), mask=False)
        al.custom(
            OP,
            A,
            B,
            OUT,
            DQ,
            DG,
            FLAG,
            SA,
            SB,
            WS,
            ext.program_id(0).to(tl.int32),
            out=dummy,
        )


def prepare(
    aq,
    bq,
    sa,
    sb,
    out,
    m,
    n,
    k,
    tiles,
    *,
    input_nz=False,
    batch_rows=False,
    prefetch=False,
):
    bm, bn, bk = tiles
    assert not batch_rows or (m % 16 == 0 and n <= 4095)
    assert not prefetch or (input_nz and batch_rows and k == 512)
    assert n % 32 == 0 and bn % 32 == 0 and k % bk == 0
    assert aq.dtype == bq.dtype == torch.int8 and sa.dtype == sb.dtype == torch.float32
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert m > 0 and n > 0 and m * n < 2**31
    mod = importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm_w8a8_fp8")
    mp, np = triton.cdiv(m, bm) * bm, triton.cdiv(n, bn) * bn
    cores = min(mod._cube_core_count(), mp // bm * (np // bn))
    if mp == m and aq.is_contiguous():
        aa = aq
    else:
        aa = torch.zeros((mp, k), device=aq.device, dtype=torch.int8)
        aa[:m] = aq
    if np == n:
        bb = bq.T.contiguous()
    else:
        bb = torch.zeros((np, k), device=aq.device, dtype=torch.int8)
        bb[:n] = bq.T
    if input_nz:
        local_bytes = (bm + bn) * k + n * 8 + bm * bn * 2 + (bm // 16 + 1) * 512
        am = bm + (16 if bm % 32 == 0 and local_bytes + 16 * k <= 512 * 1024 else 0)
        az = torch.zeros((mp // bm, am, k), device=aq.device, dtype=torch.int8)
        az[:, :bm] = aa.reshape(mp // bm, bm, k)
        aa = (
            az.reshape(mp // bm, am // 16, 16, k // 32, 32)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        bb = (
            bb.reshape(np // bn, bn // 16, 16, k // 32, 32)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
    ss = torch.zeros(mp, device=aq.device, dtype=torch.float32)
    ss[:m] = sa
    sb = sb.contiguous()
    bound = float(k * 128 * 128)
    am, bm_scale = ss.abs().amax(), sb.abs().amax()
    ref = torch.minimum(am, 30000.0 / (bound * bm_scale)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    ratio, scaled_b = ss / ref, sb * ref
    rm, min_half = ratio.abs().amax(), 2.0**-14
    # Conservative range/error checks. Tiny results may round to zero only
    # when their absolute bound remains below the existing accuracy budget.
    valid = (
        torch.isfinite(ss).all()
        & torch.isfinite(sb).all()
        & (ss >= 0).all()
        & (sb >= 0).all()
        & (rm <= 60000)
        & (scaled_b.abs().amax() * bound <= 30001)
        & (am * bound <= torch.finfo(torch.float32).max / 2)
        & ((ratio.abs() >= min_half) | (ss.abs() * bm_scale * bound <= 0.01)).all()
        & (
            (scaled_b.abs() >= min_half)
            | (rm * min_half <= 0.01)
            | (sb.abs() * am * bound <= 0.01)
        ).all()
    )
    flag = valid.to(torch.int32).reshape(1)
    dq = mod._pack_deq_u64(scaled_b)
    nd = torch.diag_embed(ratio.to(torch.float16).reshape(mp // bm, bm // 16, 16))
    zero = torch.zeros((mp // bm, 1, 16, 16), device=aq.device, dtype=torch.float16)
    dg = torch.cat((zero, nd), dim=1).contiguous()
    workspace = torch.empty((cores * bm * bn,), device=aq.device, dtype=torch.int32)
    name, revision = _register(
        m, n, k, mp, np, bm, bn, bk, cores, input_nz, batch_rows, prefetch
    )
    args = (aa, bb, out, dq, dg, flag, ss, sb, workspace, name, revision)

    def call():
        return _mm_aic_kernel[(cores,)](*args, num_warps=1)

    call.prepared_args = args
    return call, {
        "path": "aic_fpbuffer",
        "input_layout": "nz" if input_nz else "nd",
        "batch_row_scale": batch_rows,
        "prefetch_input": prefetch,
        "tiles": list(tiles),
        "kernel_count": 1,
        "workspace_bytes": workspace.numel() * workspace.element_size(),
        "range_guard": "device",
        "scalar_fallback": "fp32_axes_then_bf16",
        "scale_format": "packed_fp16_diagonal_and_uint64_dequant",
    }
